"""Genera data/digits.csv a partir del dataset digits de scikit-learn (8x8 px, 10 clases)."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits

OUT = Path(__file__).resolve().parents[1] / "data" / "digits.csv"

digits = load_digits(as_frame=True)
df = digits.frame.rename(columns={"target": "label"})
OUT.parent.mkdir(exist_ok=True)
df.to_csv(OUT, index=False)
print(f"{len(df)} filas guardadas en {OUT}")
