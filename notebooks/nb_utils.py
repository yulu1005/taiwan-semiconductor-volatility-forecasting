# nb_utils.py — notebooks 共用工具：路徑、設定、字型、配色、IC 與 HAC
import sys, pathlib                                          # 載入：系統路徑、路徑工具
ROOT = pathlib.Path(__file__).resolve().parents[1]           # 專案根目錄
sys.path.insert(0, str(ROOT / "src"))                        # 讓 notebook 能匯入 src 裡的程式
import numpy as np, pandas as pd                             # 載入：數值運算、表格
import matplotlib.pyplot as plt                              # 載入：畫圖
from common import CFG, PROC, TAB, FIG, setup_fonts, nw_mean_ci  # 載入：設定、資料夾、字型、HAC 平均
try:                                                         # 在 notebook 裡執行時
    from IPython import get_ipython                          # 取得 notebook 核心
    if get_ipython() is not None: get_ipython().run_line_magic("matplotlib", "inline")  # 圖直接顯示在 notebook 中（common.py 預設不顯示）
except ImportError: pass                                     # 一般 Python 執行時略過

NB_OUT = ROOT / "outputs" / "notebooks"; NB_OUT.mkdir(parents=True, exist_ok=True)  # notebook 輸出資料夾
STK = CFG["universe"]["stocks"]                              # 15 檔股票代號
GOLD, BM = CFG["universe"]["gold_etf"], CFG["universe"]["passive_benchmark"]  # 黃金 ETF、0050
HOLDOUT_START = pd.Timestamp(CFG["protocol"]["holdout"]["test"][0])  # 封存期起點 2025-01-01
H = CFG["schedule"]["label_horizon_days"]                    # 標籤期間 5 日
FOLDS = CFG["protocol"]["folds"]                             # 三個驗證 fold
BLUE, ORANGE, AQUA, YELLOW, GREY, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984", "#e34948"  # 固定配色

def style():                                                 # 定義：統一圖表樣式
    setup_fonts()                                            # 中文字型
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": .6, "figure.dpi": 110,
                         "savefig.dpi": 200, "axes.titlesize": 12, "axes.titlelocation": "left"})  # 低調格線、標題靠左

def save(fig, name):                                         # 定義：存圖並顯示
    fig.tight_layout(); fig.savefig(NB_OUT / name, bbox_inches="tight"); plt.show()  # 存到 outputs/notebooks 並在 notebook 顯示

def ic_by_date(df, x, y="y", min_n=10):                      # 定義：每日橫斷面 Spearman IC
    g = df[["date", x, y]].dropna().groupby("date")          # 依日期分組（去掉空值）
    out = g.apply(lambda t: t[x].rank().corr(t[y].rank()) if len(t) >= min_n else np.nan, include_groups=False)  # 至少 min_n 檔才算
    return out.dropna().sort_index()                         # 去空值並依時間排序（HAC 需要時間順序）

def hac_stats(ic, lags):                                     # 定義：HAC 與天真 t 值
    m, lo, hi, se = nw_mean_ci(ic.values, lags=lags)         # Newey–West 平均、區間、標準誤
    naive_se = ic.std() / np.sqrt(len(ic))                   # 把每天當獨立樣本的標準誤
    from scipy.stats import norm                             # 常態分配
    return {"ic_mean": m, "ic_lo": lo, "ic_hi": hi, "hac_t": m / se, "naive_t": m / naive_se,
            "hac_p": 2 * (1 - norm.cdf(abs(m / se))), "n_dates": len(ic)}  # 回傳字典
