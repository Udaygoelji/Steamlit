import os
from torchvision import datasets, transforms
from tqdm import tqdm
import random

ANIMAL_CLASSES = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']

def export_animals(root_out='data/animals', val_split=0.2, max_per_class=None, seed=42):
    random.seed(seed)
    os.makedirs(root_out, exist_ok=True)
    train_out = os.path.join(root_out, 'train')
    val_out = os.path.join(root_out, 'val')
    for d in [train_out, val_out]:
        os.makedirs(d, exist_ok=True)
        for c in ANIMAL_CLASSES:
            os.makedirs(os.path.join(d, c), exist_ok=True)

    # Download CIFAR-10
    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10(root='data/raw', train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10(root='data/raw', train=False, download=True, transform=transform)

    idx_to_class = {i: c for i, c in enumerate(trainset.classes)}

    def dump_split(split, split_name):
        per_class_counter = {c:0 for c in ANIMAL_CLASSES}
        for img, label in tqdm(split, desc=f'Exporting {split_name}'):
            class_name = idx_to_class[label]
            if class_name not in ANIMAL_CLASSES:
                continue
            if max_per_class and per_class_counter[class_name] >= max_per_class:
                continue
            pil = transforms.ToPILImage()(img)
            outdir = os.path.join(root_out, split_name, class_name)
            count = per_class_counter[class_name]
            pil.save(os.path.join(outdir, f'{class_name}_{count:05d}.jpg'))
            per_class_counter[class_name] += 1

    indices = list(range(len(trainset)))
    random.shuffle(indices)
    split_idx = int((1.0 - val_split) * len(indices))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_subset = [trainset[i] for i in train_idx]
    val_subset = [trainset[i] for i in val_idx]

    dump_split(train_subset, 'train')
    dump_split(val_subset + list(testset), 'val')

if __name__ == '__main__':
    export_animals()
