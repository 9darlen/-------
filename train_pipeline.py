import joblib
import pandas as pd
from feature_builder import FeatureBuilder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import category_encoders as ce

# 你的欄位（照你目前的設定）
TE_COLS = ['State','BankState','NAICS_Section','ApprovalFY']
OHE_COLS = ['NewExist','UrbanRural','RevLineCr','FranchiseCode_Binary','LowDoc']

# 讀資料
df = pd.read_csv("data/SBAnational.csv").dropna(subset=["MIS_Status"])
y = (df["MIS_Status"] == "CHGOFF").astype(int)
X = df.drop(columns=["MIS_Status"])

# split（建議 stratify）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 先跑 FeatureBuilder 轉換一次，用來判斷哪些是 numeric
tmp = FeatureBuilder().fit_transform(X_train)

# 把你要編碼的欄位排除掉，剩下 numeric 才加入模型
num_cols = tmp.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if c not in TE_COLS and c not in OHE_COLS]


# Step B: 混合編碼
preprocess = ColumnTransformer(
    transformers=[
        ("te", ce.TargetEncoder(cols=TE_COLS), TE_COLS),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"), OHE_COLS),
        ("num", "passthrough", num_cols)  # ✅ 加入數值欄位
    ],
    remainder="drop"
)


# Step C: 缺失 + scaler（remainder passthrough 可能含 numeric，先用 SimpleImputer+scaler 統一處理）
# 由於 preprocess 後已是「全數值矩陣」，我們直接對整個矩陣做 impute+scale
pipe = Pipeline(steps=[
    ("feat", FeatureBuilder()),
    ("encode", preprocess),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler(with_mean=False)),  # with_mean=False 對 one-hot 稀疏更安全
    ("model", RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    ))
])

pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
print("AUC:", roc_auc_score(y_test, proba))

joblib.dump(pipe, "pipeline.joblib")
print("Saved: pipeline.joblib")
