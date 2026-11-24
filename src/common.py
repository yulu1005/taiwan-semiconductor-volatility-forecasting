# common.py — 各步驟共用的路徑、設定、字型與統計工具
import pathlib, yaml, numpy as np                            # 載入：路徑、設定檔、數值運算
import matplotlib; matplotlib.use("Agg")                     # 不開視窗，直接畫到檔案
from matplotlib import font_manager, rcParams                # 載入：字型管理

ROOT = pathlib.Path(__file__).resolve().parents[1]           # 專案根目錄
CFG = yaml.safe_load(open(ROOT/"setup.yaml", encoding="utf-8"))  # 讀入 setup.yaml
PROC = ROOT/"data"/"processed"                               # 處理後資料夾
TAB, FIG = ROOT/"outputs"/"tables", ROOT/"outputs"/"figures" # 輸出表格與圖檔資料夾

def setup_fonts():                                           # 定義：設定中文字型
    for name in ["Noto Sans CJK TC", "Noto Sans CJK JP", "Microsoft JhengHei", "PingFang TC", "Heiti TC"]:  # 依序嘗試常見中文字型
        if any(name in f.name for f in font_manager.fontManager.ttflist):  # 系統有這個字型
            rcParams["font.family"] = name; break            # 就使用它
    rcParams["axes.unicode_minus"] = False                   # 負號正常顯示

def nw_mean_ci(x, lags=None):                                # 定義：Newey–West（HAC）平均值與 95% 信賴區間
    x = np.asarray(x, float); x = x[~np.isnan(x)]            # 轉成數值並去掉空值
    n = len(x); m = x.mean(); e = x - m                      # 樣本數、平均、離差
    L = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))  # 自動選落後期數
    s = e @ e / n                                            # 變異數（第 0 期）
    for l in range(1, L + 1):                                # 加上各期自我共變異
        w = 1 - l / (L + 1)                                  # Bartlett 權重
        s += 2 * w * (e[l:] @ e[:-l]) / n                    # 加權共變異
    se = np.sqrt(max(s, 0) / n)                              # 平均值的標準誤
    return m, m - 1.96 * se, m + 1.96 * se, se               # 回傳：平均、下界、上界、標準誤

VER = CFG.get("features", {}).get("active", "v1")            # 目前使用的特徵組版本
VOUT = ROOT/"outputs" if VER == "v1" else ROOT/"outputs"/VER # 版本輸出根目錄（v1 結果保留在原位，v2 另放）
VTAB, VFIG = VOUT/"tables", VOUT/"figures"                   # 版本化的表格與圖檔資料夾
for _d in (VTAB, VFIG): _d.mkdir(parents=True, exist_ok=True)  # 不存在就建立
def vproc(name):                                             # 定義：版本化的中間資料檔名
    p = pathlib.Path(name)                                   # 拆檔名
    return PROC/name if VER == "v1" else PROC/f"{p.stem}_{VER}{p.suffix}"  # v1 用原檔名，其他版本加後綴
