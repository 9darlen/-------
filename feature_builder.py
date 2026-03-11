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

        # 2) Bank / BankState 缺失 + flags
        if 'Bank' in df.columns:
            df['Is_Bank_Missing'] = df['Bank'].isna().astype(int)
            df['Bank'] = df['Bank'].fillna('Missing_Data')
            df['Is_Missing_Group'] = (df['Bank'] == 'Missing_Data').astype(int)

        if 'BankState' in df.columns:
            df['BankState'] = df['BankState'].fillna('MD')

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
        # 定義一個固定的基準日期（例如：資料集中最早日期之前的某個時間點）
        REF_DATE = pd.Timestamp('2000-01-01')
        date_cols = [
            ("ApprovalDate", "Days_Since_Appv"), 
            ("DisbursementDate", "Days_Since_Disb")
        ]
        for orig_col, new_col in date_cols:
            if orig_col in df.columns:
                # 轉成日期格式，無法轉換的會變成 NaT
                dt = pd.to_datetime(df[orig_col], errors="coerce")
                
                # 核心修正：統一減去固定基準日期 self.REF_DATE
                # 這保證了訓練與預測的一致性
                df[new_col] = (dt - self.REF_DATE).dt.days
                
                # 處理缺失值：如果日期是空的，補一個代表「未知」的數值（例如 -1 或 0）
                df[new_col] = df[new_col].fillna(-1)
                
                # 刪掉原始字串欄位
                df = df.drop(columns=[orig_col])
        # 7) 刪掉 NAICS 原碼（避免跟 NAICS_Section 重複）
        if "NAICS" in df.columns:
            df = df.drop(columns=["NAICS"], errors="ignore")
        # FranchiseCode -> FranchiseCode_Binary（有 franchise 就 1，否則 0）
        if "FranchiseCode" in df.columns and "FranchiseCode_Binary" not in df.columns:
            fc = pd.to_numeric(df["FranchiseCode"], errors="coerce")
            df["FranchiseCode_Binary"] = (fc.fillna(0) != 0).astype(int)


        return df
