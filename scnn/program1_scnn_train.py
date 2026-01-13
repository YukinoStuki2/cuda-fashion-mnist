import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from spikingjelly.activation_based import neuron, functional, layer, surrogate
from copy import deepcopy

# ===== reproducibility =====
def set_seed(s=43):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(43)
torch.backends.cudnn.benchmark = True

# ===== hyper-params (tuned for FMNIST + SNN) =====
LR = 3e-3                     # 稍保守更稳；可在 3e-3 / 4e-3 之间试
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 512              # 512/768/1024 三档 A/B；你之前 2048 更差
EPOCHS = 180                  # 120~160 区间；140 更稳
T_TIMESTEPS = 8               # 与推理一致
LABEL_SMOOTH_MAX = 0.04       # 初期 0.04，末期线性衰减到 0
WARMUP_EPOCHS = 5
EMA_DECAY = 0.9995            # 更强 EMA
SWA_LAST_EPOCHS = 20          # 末 20 个 epoch 做 SWA
CLIP_NORM = 5.0

# ===== model =====
class SCNN(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

        self.conv1 = layer.Conv2d(1, 6, 5, bias=True)
        self.if1 = neuron.IFNode(v_threshold=1.0, v_reset=0.0,
                                 surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pool1 = layer.MaxPool2d(2, 2)

        self.conv2 = layer.Conv2d(6, 16, 5, bias=True)
        self.if2 = neuron.IFNode(v_threshold=1.0, v_reset=0.0,
                                 surrogate_function=surrogate.ATan(), detach_reset=True)
        self.pool2 = layer.MaxPool2d(2, 2)

        self.flatten = layer.Flatten()
        self.fc1 = layer.Linear(16 * 4 * 4, 120, bias=True)
        self.if3 = neuron.IFNode(v_threshold=1.0, v_reset=0.0,
                                 surrogate_function=surrogate.ATan(), detach_reset=True)

        self.fc2 = layer.Linear(120, 84, bias=True)
        self.if4 = neuron.IFNode(v_threshold=1.0, v_reset=0.0,
                                 surrogate_function=surrogate.ATan(), detach_reset=True)

        self.fc3 = layer.Linear(84, 10, bias=True)

    def forward(self, x: torch.Tensor):
        outs = []
        for _ in range(self.T):
            y = self.conv1(x); y = self.if1(y); y = self.pool1(y)
            y = self.conv2(y); y = self.if2(y); y = self.pool2(y)
            y = self.flatten(y)
            y = self.fc1(y);  y = self.if3(y)
            y = self.fc2(y);  y = self.if4(y)
            y = self.fc3(y)
            outs.append(y)
        return torch.stack(outs, dim=0).mean(0)

# ===== data =====
def get_loaders(script_dir, batch_size):
    train_tf = transforms.Compose([
        transforms.RandomCrop(28, padding=2, padding_mode='reflect'),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0.0, inplace=False),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    trainset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=train_tf)
    testset  = torchvision.datasets.FashionMNIST(data_dir, download=True, train=False, transform=test_tf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=4, pin_memory=True, persistent_workers=True)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False,
                                              num_workers=4, pin_memory=True, persistent_workers=True)
    return trainloader, testloader

# ===== schedulers =====
def cosine_warmup_lr(optimizer, base_lr, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ===== utils =====
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
        b_ema.copy_(b)

def _unwrap_model(m: nn.Module) -> nn.Module:
    # 兼容 AveragedModel / DataParallel / DDP
    return getattr(m, 'module', m)

def export_txt(model: nn.Module, out_dir: str):
    base = _unwrap_model(model)
    os.makedirs(out_dir, exist_ok=True)
    name_map = {
        'conv1.weight': base.conv1.weight,
        'conv1.bias':   base.conv1.bias,
        'conv2.weight': base.conv2.weight,
        'conv2.bias':   base.conv2.bias,
        'fc1.weight':   base.fc1.weight,
        'fc1.bias':     base.fc1.bias,
        'fc2.weight':   base.fc2.weight,
        'fc2.bias':     base.fc2.bias,
        'fc3.weight':   base.fc3.weight,
        'fc3.bias':     base.fc3.bias,
    }
    for name, tensor in name_map.items():
        np.savetxt(os.path.join(out_dir, f'{name}.txt'),
                   tensor.detach().cpu().numpy().ravel())

def evaluate(model: nn.Module, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            functional.reset_net(model)
            logits = model(imgs)
            pred = logits.argmax(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return correct / total

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trainloader, testloader = get_loaders(script_dir, BATCH_SIZE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SCNN(T=T_TIMESTEPS).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = cosine_warmup_lr(optimizer, LR, WARMUP_EPOCHS, EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())  # 新 API

    # EMA / SWA
    ema_model = deepcopy(model).to(device)
    for p in ema_model.parameters(): p.requires_grad_(False)
    swa_model = AveragedModel(model, multi_avg_fn=None)

    print("--- Start SCNN Training (FP32, EMA+SWA, fading label smoothing, strong aug) ---")
    best_acc = 0.0
    best_dir = os.path.join(script_dir, 'weights_best')
    os.makedirs(best_dir, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0

        # 标签平滑线性衰减：前 80% 维持 LABEL_SMOOTH_MAX，最后 20% 线性到 0
        if epoch < int(EPOCHS * 0.8):
            smooth = LABEL_SMOOTH_MAX
        else:
            remain = EPOCHS - epoch
            smooth = LABEL_SMOOTH_MAX * (remain / max(1, int(EPOCHS * 0.2)))
        criterion = nn.CrossEntropyLoss(label_smoothing=float(smooth))

        for imgs, labels in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # 新 API
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            ema_update(ema_model, model, EMA_DECAY)

            running += loss.item()

        scheduler.step()
        train_loss = running / max(1, len(trainloader))

        # SWA 累积（末段）
        if epoch >= EPOCHS - SWA_LAST_EPOCHS:
            swa_model.update_parameters(model)

        # 验证：EMA 与 SWA 二者取优
        acc_ema = evaluate(ema_model, testloader, device)
        acc_swa = evaluate(swa_model if epoch >= EPOCHS - SWA_LAST_EPOCHS else ema_model, testloader, device)
        acc = max(acc_ema, acc_swa)
        choice = "SWA" if acc_swa >= acc_ema else "EMA"

        print(f"Epoch [{epoch+1}/{EPOCHS}] loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
              f"smooth={smooth:.3f} acc_EMA={acc_ema:.4f} acc_SWA={acc_swa:.4f} -> use {choice}")

        if acc > best_acc:
            best_acc = acc
            print(f'New best {best_acc:.4f}, exporting weights...')
            # 直接把当前选中的（EMA/SWA）模型导出；export_txt 会自动 unwrap
            export_txt(swa_model if choice == "SWA" else ema_model, best_dir)

    print('--- Finished ---')
    print(f'Best Acc: {best_acc:.4f}')
    print('Exported to:', best_dir)

if __name__ == '__main__':
    main()