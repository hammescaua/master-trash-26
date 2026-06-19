"""
scripts/split_dataset.py

Divide o dataset bruto em treino (70%), validação (15%) e teste (15%).
Cria a estrutura dataset/train|val|test/<classe>/.
"""

import os
import shutil
import random
from pathlib import Path

CLASSES   = ["paper", "plastic", "metal", "organic"]
RAW_DIR   = Path("dataset/raw")
SPLITS    = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED      = 42


def split_class(class_name: str):
    src = RAW_DIR / class_name
    images = sorted(src.glob("*.jpg"))

    if not images:
        print(f"[AVISO] Nenhuma imagem encontrada em {src}")
        return

    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    groups = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, files in groups.items():
        dest = Path("dataset") / split_name / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, dest / f.name)
        print(f"  {split_name}/{class_name}: {len(files)} imagens")


def main():
    random.seed(SEED)
    print("=== Divisão do Dataset ===")
    for cls in CLASSES:
        print(f"\n[{cls}]")
        split_class(cls)
    print("\nDivisão concluída.")


if __name__ == "__main__":
    main()
