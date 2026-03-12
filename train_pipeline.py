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
num_cols = [c for c in num_cols if c not in TE_COLS and c not in OHE_COLS and c != 'MIS_Status']


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
    ("model", RandomForestClassifier(random_state=42,n_jobs=-1))
])

# Step D: 定義搜尋參數空間
# 注意：參數名稱必須以 Pipeline 中的步驟名稱開頭，並用兩個底線連接
param_grid = {
    'model__n_estimators': [100, 400],
    'model__max_depth': [10, 20, None],
    'model__min_samples_split': [2, 5],
    'model__class_weight': ['balanced', None] # 處理資料不平衡
}

# Step E: 建立 GridSearchCV
# n_jobs=-1 使用所有 CPU 核心，cv=5 代表五折交叉驗證
grid_search = GridSearchCV(
    pipe, 
    param_grid, 
    cv=5, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=2
)

# Step F: 開始搜救（這會花一點時間，因為它在跑每一 row 參數組合）
print("開始 Grid Search...")
grid_search.fit(X_train, y_train)

# Step G: 輸出結果
print(f"最佳參數: {grid_search.best_params_}")
print(f"最佳交叉驗證 AUC: {grid_search.best_score_:.4f}")

# 使用最佳模型預測測試集
best_model = grid_search.best_estimator_
proba = best_model.predict_proba(X_test)[:, 1]
final_auc = roc_auc_score(y_test, proba)
print(f"測試集最終 AUC: {final_auc:.4f}")

# 獲取模型特徵重要性
model = pipe.named_steps['model']
# 這裡需要從 preprocess 取得轉換後的欄位名稱
feature_names = pipe.named_steps['encode'].get_feature_names_out()

importance = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
print(importance.sort_values(by='importance', ascending=False).head(10))

joblib.dump(best_model, "best_pipeline.joblib")
print("Saved: pipeline.joblib")
