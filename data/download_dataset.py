from pathlib import Path
from shutil import copy2

import kagglehub


DATASET_NAME = "psparks/instacart-market-basket-analysis"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "instacart"


def main() -> None:
    """Baixa o dataset Instacart e copia os arquivos para data/raw."""
    downloaded_path = Path(kagglehub.dataset_download(DATASET_NAME))
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for csv_file in downloaded_path.glob("*.csv"):
        copy2(csv_file, RAW_DATA_DIR / csv_file.name)

    print(f"Dataset baixado em: {downloaded_path}")
    print(f"Arquivos copiados para: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
