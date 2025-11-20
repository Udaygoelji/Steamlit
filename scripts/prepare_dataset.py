from src.dataset import export_animals

if __name__ == '__main__':
    export_animals(root_out='data/animals', val_split=0.2, max_per_class=None, seed=42)
    print('Dataset prepared at data/animals/')
