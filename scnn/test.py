import os, numpy as np, torch, torchvision
from torch import nn
import export_fix
def load_txt(path):
    return np.loadtxt(path)

def build_data(batch=256):
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    ds = torchvision.datasets.FashionMNIST("data", download=True, train=False, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False)
    imgs, labels = next(iter(loader))
    return imgs, labels

def try_orders(weights_dir):
    # Expected shapes (wide T=2)
    conv1_w = load_txt(os.path.join(weights_dir, "conv1.weight.txt"))  # 8*1*5*5
    conv1_b = load_txt(os.path.join(weights_dir, "conv1.bias.txt"))    # 8
    conv2_w = load_txt(os.path.join(weights_dir, "conv2.weight.txt"))  # 32*8*5*5
    conv2_b = load_txt(os.path.join(weights_dir, "conv2.bias.txt"))    # 32
    fc1_w   = load_txt(os.path.join(weights_dir, "fc1.weight.txt"))    # 192*512
    fc1_b   = load_txt(os.path.join(weights_dir, "fc1.bias.txt"))      # 192
    fc2_w   = load_txt(os.path.join(weights_dir, "fc2.weight.txt"))    # 128*192
    fc2_b   = load_txt(os.path.join(weights_dir, "fc2.bias.txt"))      # 128
    fc3_w   = load_txt(os.path.join(weights_dir, "fc3.weight.txt"))    # 10*128
    fc3_b   = load_txt(os.path.join(weights_dir, "fc3.bias.txt"))      # 10

    imgs, labels = build_data(batch=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    imgs = imgs.to(device)

    results = []

    # conv weight ordering: [O,I,K,K] vs [I,O,K,K]
    def reshape_conv(arr, O, I, K, mode):
        if mode == 'OI':  # [O,I,K,K]
            npw = arr.reshape(O, I, K, K)
        elif mode == 'IO':  # [I,O,K,K] -> transpose to [O,I,K,K]
            npw = arr.reshape(I, O, K, K).transpose(1, 0, 2, 3)
        else:
            raise ValueError(mode)
        return torch.tensor(npw, dtype=torch.float32, device=device)

    # FC ordering: row-major [O,I] vs [I,O]
    def reshape_fc(arr, O, I, mode):
        if mode == 'OI':
            npw = arr.reshape(O, I)
        elif mode == 'IO':
            npw = arr.reshape(I, O).transpose(1, 0)
        else:
            raise ValueError(mode)
        return torch.tensor(npw, dtype=torch.float32, device=device)

    conv1_modes = ['OI','IO']
    conv2_modes = ['OI','IO']
    fc_modes    = ['OI','IO']

    for m1 in conv1_modes:
        W1 = reshape_conv(conv1_w, 8, 1, 5, m1)
        b1 = torch.tensor(conv1_b, dtype=torch.float32, device=device)
        conv1 = nn.Conv2d(1, 8, 5, bias=True).to(device)
        with torch.no_grad():
            conv1.weight.copy_(W1)
            conv1.bias.copy_(b1)

        for m2 in conv2_modes:
            W2 = reshape_conv(conv2_w, 32, 8, 5, m2)
            b2 = torch.tensor(conv2_b, dtype=torch.float32, device=device)
            conv2 = nn.Conv2d(8, 32, 5, bias=True).to(device)
            with torch.no_grad():
                conv2.weight.copy_(W2)
                conv2.bias.copy_(b2)

            flatten = nn.Flatten()
            mp2 = nn.MaxPool2d(2)

            for m3 in fc_modes:
                Wfc1 = reshape_fc(fc1_w, 192, 512, m3)
                bfc1 = torch.tensor(fc1_b, dtype=torch.float32, device=device)
                fc1  = nn.Linear(512, 192, bias=True).to(device)
                with torch.no_grad():
                    fc1.weight.copy_(Wfc1)
                    fc1.bias.copy_(bfc1)

                for m4 in fc_modes:
                    Wfc2 = reshape_fc(fc2_w, 128, 192, m4)
                    bfc2 = torch.tensor(fc2_b, dtype=torch.float32, device=device)
                    fc2  = nn.Linear(192, 128, bias=True).to(device)
                    with torch.no_grad():
                        fc2.weight.copy_(Wfc2)
                        fc2.bias.copy_(bfc2)

                    for m5 in fc_modes:
                        Wfc3 = reshape_fc(fc3_w, 10, 128, m5)
                        bfc3 = torch.tensor(fc3_b, dtype=torch.float32, device=device)
                        fc3  = nn.Linear(128, 10, bias=True).to(device)
                        with torch.no_grad():
                            fc3.weight.copy_(Wfc3)
                            fc3.bias.copy_(bfc3)

                        # forward (analog, non-spiking) just to test weight orders
                        with torch.no_grad():
                            x = imgs
                            y = conv1(x); y = torch.relu(y)
                            y = mp2(y)
                            y = conv2(y); y = torch.relu(y)
                            y = mp2(y)
                            y = flatten(y)
                            y = fc1(y);  y = torch.relu(y)
                            y = fc2(y);  y = torch.relu(y)
                            logits = fc3(y)
                            pred = logits.argmax(1).cpu()
                            acc = (pred == labels[:pred.size(0)]).float().mean().item()

                        results.append(((m1, m2, m3, m4, m5), acc))

    results.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 order combinations by accuracy:")
    for top in results[:10]:
        print("order (conv1, conv2, fc1, fc2, fc3) =", top[0], "acc=", top[1])

if __name__ == "__main__":
    weights_dir = r"F:\C\cuda\two\three\weight"
    # try_orders(weights_dir)
    export_fix.verify_txt_as_cnn(weights_dir)