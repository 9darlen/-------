import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CURRENCY_COLS = ['SBA_Appv','GrAppv','ChgOffPrinGr','BalanceGross','DisbursementGross']
LOG_COLS = ['BalanceGross','SBA_Appv','GrAppv','DisbursementGross']

class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # 1) 金額欄位清理成 numeric
        for c in CURRENCY_COLS:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(r'[$, ]', '', regex=True)
                df[c] = pd.to_numeric(df[c], errors='coerce')


        # 3) log1p 金額（先 clip 到 >=0）
        for c in LOG_COLS:
            if c in df.columns:
                df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").clip(lower=0))

        # 4) NAICS -> NAICS_Section（取前兩碼）
        if "NAICS" in df.columns and "NAICS_Section" not in df.columns:
            df["NAICS_Section"] = df["NAICS"].astype(str).str[:2]
            df["NAICS_Section"] = df["NAICS_Section"].replace({
                "0": "Unknown", "00": "Unknown",
                "na": "Unknown", "n": "Unknown",
                "nan": "Unknown", "None": "Unknown", "": "Unknown"
            })

        # 5) RevLineCr / LowDoc 缺失補 'nan'
        for c in ["RevLineCr", "LowDoc"]:
            if c in df.columns:
                df[c] = df[c].fillna("nan")

        
        # 6) ApprovalDate / DisbursementDate -> Days_Since_...
        # 定義基準日期
        REF_DATE = pd.Timestamp('1970-01-01') 
        date_cols = [
            ("ApprovalDate", "Days_Since_Appv"), 
            ("DisbursementDate", "Days_Since_Disb")
        ]
        
        for orig_col, new_col in date_cols:
            if orig_col in df.columns:
                # A. 強制指定格式解析 28-Feb-97
                # %y 是兩位數年份，%b 是月份縮寫
                dt = pd.to_datetime(df[orig_col], format='%d-%b-%y', errors="coerce")
                
                # B. 修正世紀問題 (Pandas 預設 68-99 為 19xx, 00-67 為 20xx)
                # 假設 SBA 貸款不會出現在未來（例如 2026 年之後）
                future_mask = dt.dt.year > 2026
                dt.loc[future_mask] -= pd.DateOffset(years=100)
                
                # C. 計算天數（轉換為整數列 row）
                df[new_col] = (dt - REF_DATE).dt.days
                
                # D. 暫時補 -1 (稍後在 Pipeline 中可用 SimpleImputer 針對 Column 補中位數)
                # 這裡保留 -1 是為了標記原本就是空值的資料列
                df[new_col] = df[new_col].fillna(-1)
                
                # 刪除原始字串欄位
                df = df.drop(columns=[orig_col])
        # 7) 刪掉 NAICS 原碼（避免跟 NAICS_Section 重複）
        if "NAICS" in df.columns:
            df = df.drop(columns=["NAICS"], errors="ignore")
        # FranchiseCode -> FranchiseCode_Binary（有 franchise 就 1，否則 0）
        if "FranchiseCode" in df.columns and "FranchiseCode_Binary" not in df.columns:
        # 轉成數值，無法轉的變 NaN
         fc = pd.to_numeric(df["FranchiseCode"], errors="coerce").fillna(0)
    
        # 0 或 1 都會被判定為 False (0)，其他數值（例如 2、3、4...）都會被判定為 True (1)
         df["FranchiseCode_Binary"] = (~fc.isin([0, 1])).astype(int)
    
         # 既然已經轉成 Binary 特徵，建議刪除原始欄位以免干擾模型
         df = df.drop(columns=["FranchiseCode"])


        return df
