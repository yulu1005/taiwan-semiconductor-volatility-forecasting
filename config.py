"""集中管理專案路徑與參數。"""

from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
ADJUSTED_CLOSE_PATH = EXTERNAL_DATA_DIR / "adjusted_close.csv"
ADJUSTED_PRICE_OVERRIDES_PATH = EXTERNAL_DATA_DIR / "adjusted_price_overrides.csv"
RAW_DATA_DIR = DATA_DIR / "raw"
PRICE_CACHE_DIR = RAW_DATA_DIR / ".price_cache"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROJECT_DIR / "features"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

STOCK_CODES = [
    "2303", "2330", "2337", "2340", "2344",
    "2371", "2408", "2434", "2455", "3016",
    "3532", "4919", "5222", "6770", "8028",
]
LOOKBACK_DAYS = 20
FORECAST_HORIZON = 5
ANNUAL_TRADING_DAYS = 252
REBALANCE_DAYS = 5

START_DATE = date(2016, 11, 1)
END_DATE = date(2026, 11, 1)
MIN_VALID_DAYS = 500
MIN_COVERAGE = 0.95
PRICE_JUMP_THRESHOLD = 0.20
REFERENCE_STOCK = "2330"
