import argparse, os
import numpy as np

NAMES = [
    ('conv1.weight', (6, 1, 5, 5)), ('conv1.bias', (6,)),
    ('conv2.weight', (16, 6, 5, 5)), ('conv2.bias', (16,)),
    ('fc1.weight', (120, 256)), ('fc1.bias', (120,)),
    ('fc2.weight', (84, 120)),  ('fc2.bias', (84,)),
    ('fc3.weight', (10, 84)),   ('fc3.bias', (10,)),
]

def load_one_dir(d):
    tensors = {}
    for name, shape in NAMES:
        p = os.path.join(d, f'{name}.txt')
        arr = np.loadtxt(p, dtype=np.float32).reshape(shape)
        tensors[name] = arr
    return tensors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help='One or more weight dirs each containing 10 txt files')
    ap.add_argument('--output', required=True, help='Output dir for the soup weights (10 txt files)')
    args = ap.parse_args()

    assert len(args.inputs) >= 2, 'Need at least two weight dirs to soup.'
    soups = [load_one_dir(d) for d in args.inputs]

    os.makedirs(args.output, exist_ok=True)
    for name, shape in NAMES:
        avg = sum(s[name] for s in soups) / float(len(soups))
        np.savetxt(os.path.join(args.output, f'{name}.txt'), avg.reshape(-1))
    print('Soup exported to:', args.output)

if __name__ == '__main__':
    main()