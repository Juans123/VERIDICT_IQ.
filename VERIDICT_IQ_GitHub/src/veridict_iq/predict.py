from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import add_adjusted_features


def predict_file(model_path: Path, input_path: Path, output_path: Path) -> None:
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)
    adjusted = add_adjusted_features(data)
    probability = model.predict_proba(adjusted)[:, 1]
    result = data.copy()
    result["probabilidad_favorable"] = probability
    result.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predict_file(args.model, args.input, args.output)
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
