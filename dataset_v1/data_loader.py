# Auto-generated helper to reload saved dataframes
from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path("dataset_v1")

def _read_manifest():
    with open(BASE_DIR / "manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)

def _reconstruct_dtypes(df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    # Reconstruct categoricals saved as strings
    for col in entry.get("categories", []):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df

def load_one(name: str) -> pd.DataFrame:
    manifest = _read_manifest()
    if name not in manifest["frames"]:
        raise KeyError(f"{name} not found in manifest. Available: {list(manifest['frames'])}")
    entry = manifest["frames"][name]
    parts = [BASE_DIR / p for p in entry["chunks"]]
    if not parts:
        # empty dataframe with correct columns/dtypes
        cols = list(entry["columns"].keys())
        df = pd.DataFrame(columns=cols)
        for c, dt in entry["columns"].items():
            try:
                df[c] = df[c].astype(dt)
            except Exception:
                pass
        return df
    # Concatenate
    dfs = [pd.read_parquet(p) for p in parts]
    out = pd.concat(dfs, ignore_index=True)
    out = _reconstruct_dtypes(out, entry)
    return out

def load_frames() -> dict:
    manifest = _read_manifest()
    return { name: load_one(name) for name in manifest["frames"].keys() }
