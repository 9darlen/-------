import joblib
import pandas as pd
from feature_builder import FeatureBuilder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer,make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import category_encoders as ce

# 1. 欄位設定
TE_COLS = ['State','BankState','NAICS_Section','ApprovalFY']
OHE_COLS = ['NewExist','UrbanRural','RevLineCr','FranchiseCode_Binary','LowDoc']


# 2. 讀取資料並「抽樣」 (解決執行過久與環境相容問題)
print("正在讀取資料...")
df = pd.read_csv("data/SBAnational.csv").dropna(subset=["MIS_Status"])

#刪掉不重要的欄位
delete_cols = ['LoanNr_ChkDgt', 'Name', 'City', 'Zip', 'Bank', 'ChgOffPrinGr','ChgOffDate']
df = df.drop(columns=delete_cols)

#缺失值處理
#state以n取代缺失值
df['State'] = df['State'].fillna('n')   

#刪除NewExist缺失的資料
df = df.dropna(subset=['NewExist'])

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
        ("num_std", StandardScaler(), make_column_selector(dtype_include=["number"]))
    ],
    remainder="drop"
)

#加入elasticnet的特徵選擇器
# 使用 LogisticRegressionCV 替代 ElasticNetCV (專為分類設計)
# penalty='elasticnet' 搭配 solver='saga'
selector_model = LogisticRegressionCV(
    l1_ratios=[.1, .5, .7, .9, .95, .99, 1], # 尋找最佳 L1/L2 比例
    penalty='elasticnet',
    solver='saga', 
    cv=5,
    random_state=42,
    max_iter=5000,
    n_jobs=-1 # 使用多核心加速
)

#  定義特徵選取器
# 它會自動抓取所有係數 (coef_) 不為 0 的欄位 (column)
feature_selector = SelectFromModel(selector_model, prefit=False)


# 6. Step C: 建立 Pipeline 
pipe = Pipeline(steps=[
    ("feat", FeatureBuilder()),
    ("impute", SimpleImputer(strategy="median")),
    ("encode", preprocess),
    ("scale", StandardScaler(with_mean=False)),
    ("selector", feature_selector),
    ("model", RandomForestClassifier(
        n_estimators=200,      # 直接設定參數，不搜尋
        max_depth=15, 
        class_weight={0: 1, 1: 10}, 
        random_state=42,
        n_jobs=-1
    )),
    ("model", RandomForestClassifier(
        n_estimators=200,      # 直接設定參數，不搜尋
        max_depth=15, 
        class_weight={0: 1, 1: 10}, 
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

# train.py 的最後面
import joblib
print("嘗試重新載入模型...")
test_load = joblib.load("best_pipeline.joblib")
print("載入成功！代表環境沒問題。")