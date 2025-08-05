# Dataset Export (v1)

This folder was auto-generated to store four dataframes in < 100 MB parquet chunks.

## Structure
- `manifest.json` — schema, categories, and chunk lists
- `data_loader.py` — helpers to reload
- one subfolder per dataframe with chunked parquet files

## Usage
```python
from data_loader import load_frames, load_one

# Load all as a dict
dfs = load_frames()
df_all = dfs["df_all_countries"]

# Or load a single one
df_active_tags = load_one("df_active_countries_tags")
```

## Frames
- **df_all_countries**: {chunks: 6, rows: 14640001, cols: 10}
- **df_active_countries**: {chunks: 4, rows: 8466596, cols: 10}
- **df_all_countries_tags**: {chunks: 5, rows: 11135197, cols: 12}
- **df_active_countries_tags**: {chunks: 3, rows: 6356047, cols: 12}