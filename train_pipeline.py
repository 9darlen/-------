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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import category_encoders as ce

# 1. 欄位設定
TE_COLS = ['State','BankState','NAICS_Section','ApprovalFY']
OHE_COLS = ['NewExist','UrbanRural','RevLineCr','FranchiseCode_Binary','LowDoc']

# 2. 讀取資料並「抽樣」 (解決執行過久與環境相容問題)
print("正在讀取資料...")
df = pd.read_csv("data/SBAnational.csv").dropna(subset=["MIS_Status"])

# --- 💡 關鍵修改：先抽樣 10,000 row 確保快速產出相容模型 ---
df = df.sample(n=10000, random_state=42) 

y = (df["MIS_Status"] == "CHGOFF").astype(int)
X = df.drop(columns=["MIS_Status"])

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 判斷 Numeric 欄位
tmp = FeatureBuilder().fit_transform(X_train)
num_cols = tmp.select_dtypes(include="number").columns.tolist()
num_cols = [c for c in num_cols if c not in TE_COLS and c not in OHE_COLS and c != 'MIS_Status']

# 5. Step B: 混合編碼
preprocess = ColumnTransformer(
    transformers=[
        ("te", ce.TargetEncoder(cols=TE_COLS), TE_COLS),
        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"), OHE_COLS),
        ("num", "passthrough", num_cols)
    ],
    remainder="drop"
)

# 6. Step C: 建立 Pipeline (移除 GridSearchCV，直接放入參數)
pipe = Pipeline(steps=[
    ("feat", FeatureBuilder()),
    ("encode", preprocess),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler(with_mean=False)),
    ("model", RandomForestClassifier(
        n_estimators=200,      # 直接設定參數，不搜尋
        max_depth=15, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    ))
])

# 7. 開始訓練
print("開始快速訓練模型...")
pipe.fit(X_train, y_train)

# 8. 評估測試集
proba = pipe.predict_proba(X_test)[:, 1]
print(f"測試集 AUC: {roc_auc_score(y_test, proba):.4f}")

# 9. 獲取特徵重要性 (修正原代碼中從 pipe 拿 model 的邏輯)
model_step = pipe.named_steps['model']
feature_names = pipe.named_steps['encode'].get_feature_names_out()

importance = pd.DataFrame({'feature': feature_names, 'importance': model_step.feature_importances_})
print("\n前 10 大重要特徵:")
print(importance.sort_values(by='importance', ascending=False).head(10))

# 10. 儲存模型 (確保檔名與 app.py 一致)
joblib.dump(pipe, "best_pipeline.joblib")
print("\n✅ 模型已成功儲存為 best_pipeline.joblib")