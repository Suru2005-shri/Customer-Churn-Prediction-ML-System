import requests
import pandas as pd
from .config import DATA_URL, RAW_DATA_PATH, DATA_DIR


def download_dataset(force: bool = False) -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists() and not force:
        return str(RAW_DATA_PATH)
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    RAW_DATA_PATH.write_bytes(response.content)
    return str(RAW_DATA_PATH)


def load_data() -> pd.DataFrame:
    path = download_dataset()
    df = pd.read_csv(path)
    return df
