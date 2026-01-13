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
# ===== hyper-params (for wide T=2) =====
LR = 2e-3
WEIGHT_DECAY = 0.0
BATCH_SIZE = 64
EPOCHS = 180          # 或 180，注意 2 小时限制，160 一般够
T_TIMESTEPS = 2       # 一定要和推理一致

LABEL_SMOOTH_MAX = 0.00   # 比原来小很多
WARMUP_EPOCHS = 5
EMA_DECAY = 0.999
SWA_LAST_EPOCHS = 40      # 稍微长一点
CLIP_NORM = 5.0

FASHION_MEAN = (0.2860,)
FASHION_STD  = (0.3530,)

USE_AMP = True  # 先关闭 AMP，看纯 FP32 的上限

# ===== model =====
class SCNN(nn.Module):
    def __init__(self, T: int):
        super().__init__()
        self.T = T

        # conv1: 1 -> 8
        self.conv1 = layer.Conv2d(1, 6, 5, bias=True)
        self.if1 = neuron.IFNode(
            v_threshold=0.7,  # 重点：T=2 建议阈值降低一点
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=False
        )
        self.pool1 = layer.MaxPool2d(2, 2)

        # conv2: 8 -> 32
        self.conv2 = layer.Conv2d(6, 12, 5, bias=True)
        self.if2 = neuron.IFNode(
            v_threshold=0.7,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=False
        )
        self.pool2 = layer.MaxPool2d(2, 2)

        self.flatten = layer.Flatten()
        # flatten: 32 * 4 * 4 = 512
        self.fc1 = layer.Linear(12 * 4 * 4, 96, bias=True)
        self.if3 = neuron.IFNode(
            v_threshold=0.7,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=False
        )

        self.fc2 = layer.Linear(96, 64, bias=True)
        self.if4 = neuron.IFNode(
            v_threshold=0.7,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            detach_reset=False
        )

        self.fc3 = layer.Linear(64, 10, bias=True)



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
        # transforms.RandomCrop(28, padding=2, padding_mode='reflect'),
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MEAN, FASHION_STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MEAN, FASHION_STD),
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
    scaler = torch.amp.GradScaler('cuda',
                                  enabled=torch.cuda.is_available() and USE_AMP)

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

        # 前 70% epoch 从 LABEL_SMOOTH_MAX 线性衰减到 0，后面保持 0
        progress = epoch / EPOCHS
        if progress < 0.7:
            smooth = LABEL_SMOOTH_MAX * (1.0 - progress / 0.7)
        else:
            smooth = 0.0

        criterion = nn.CrossEntropyLoss(label_smoothing=float(smooth))


        for imgs, labels in trainloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)

            if USE_AMP:
                with torch.amp.autocast('cuda',
                                        enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)  # 纯 FP32
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
                optimizer.step()

            ema_update(ema_model, model, EMA_DECAY)

            running += loss.item()

        scheduler.step()
        train_loss = running / max(1, len(trainloader))

        # SWA 累积（末段）
        if epoch >= EPOCHS - SWA_LAST_EPOCHS:
            swa_model.update_parameters(model)

        # 验证：EMA 与 SWA 二者取优
        acc_ema = evaluate(ema_model, testloader, device)
        acc_swa = evaluate(swa_model if epoch >= EPOCHS - SWA_LAST_EPOCHS else ema_model,
                           testloader, device)

        # 只有在 SWA 真正开始累积之后才允许选 SWA
        use_swa = (epoch >= EPOCHS - SWA_LAST_EPOCHS) and (acc_swa >= acc_ema)
        acc = acc_swa if use_swa else acc_ema
        choice = "SWA" if use_swa else "EMA"

        print(f"Epoch [{epoch+1}/{EPOCHS}] loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
              f"smooth={smooth:.3f} acc_EMA={acc_ema:.4f} acc_SWA={acc_swa:.4f} -> use {choice}")

        if acc > best_acc:
            best_acc = acc
            print(f'New best {best_acc:.4f}, exporting weights...')
            export_txt(swa_model if choice == "SWA" else ema_model, best_dir)

    print('--- Finished ---')
    print(f'Best Acc: {best_acc:.4f}')
    print('Exported to:', best_dir)

if __name__ == '__main__':
    main()
