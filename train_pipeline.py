import joblib
import pandas as pd
import sys
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

# 檢查是否為測試模式（由 GitHub Actions 觸發）
is_test = any("test" in arg.lower() for arg in sys.argv)

# 1. 欄位設定
TE_COLS = ['State','BankState','NAICS_Section','ApprovalFY']
OHE_COLS = ['NewExist','UrbanRural','RevLineCr','FranchiseCode_Binary','LowDoc']


# 2. 讀取資料
print("正在讀取資料...")
if is_test:
    print("⚠️ 測試模式：生成模擬數據以跳過大型 CSV 讀取...")
    # 建立一個包含所有必要欄位 (column) 的假資料
    dummy_data = {
        'State': ['CA']*10, 'BankState': ['CA']*10, 'ApprovalFY': ['2006']*10,
        'NAICS': ['236115']*10, 'NewExist': [1.0]*10, 'UrbanRural': [1]*10,
        'RevLineCr': ['N']*10, 'LowDoc': ['N']*10, 'FranchiseCode': ['0']*10,
        'SBA_Appv': ['$10,000']*10, 'GrAppv': ['$10,000']*10, 
        'DisbursementGross': ['$10,000']*10, 'BalanceGross': ['$0']*10,
        'ChgOffPrinGr': ['$0']*10, 'ApprovalDate': ['1-Jan-06']*10,
        'MIS_Status': ['P I F']*10
    }
    df = pd.DataFrame(dummy_data)
else:
    # 正式訓練時才讀取實際檔案
    df = pd.read_csv("data/SBAnational.csv").dropna(subset=["MIS_Status"])


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

# 5. 預處理設定
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
    l1_ratios=[.1, .5, .99], # 尋找最佳 L1/L2 比例
    penalty='elasticnet',
    solver='saga', 
    cv=3,
    random_state=42,
    tol=0.01,
    max_iter=1000,
    verbose=3,#顯示繁瑣進度
    n_jobs=-1 # 使用多核心加速
)

#  定義特徵選取器
# 它會自動抓取所有係數 (coef_) 不為 0 的欄位 (column)
feature_selector = SelectFromModel(selector_model, prefit=False)


# 6. Step C: 建立 Pipeline 
pipe = Pipeline(steps=[
    ("feat", FeatureBuilder()),
    ("encode", preprocess),
    ("scale", StandardScaler(with_mean=False)),
    ("selector", feature_selector),
    ("model", RandomForestClassifier(
        n_estimators=200,      # 直接設定參數，不搜尋
        max_depth=15, 
        class_weight={0: 1, 1: 10}, 
        random_state=42,
        verbose=3,
        n_jobs=-1
    )),
])

# 7. 開始訓練
print("開始快速訓練模型...")
pipe.fit(X_train, y_train)

# 8. 評估測試集
proba = pipe.predict_proba(X_test)[:, 1]
print(f"測試集 AUC: {roc_auc_score(y_test, proba):.4f}")

# 9. 獲取特徵重要性 (修正原代碼中從 pipe 拿 model 的邏輯)
# 1. 取得 Pipeline 中的各個組件
model_step = pipe.named_steps['model']
selector_step = pipe.named_steps['selector']
encode_step = pipe.named_steps['encode']

# 2. 取得「編碼後」的所有欄位名稱 (這是篩選前的全名清單)
# 注意：這裡抓的是 encode 站產出的所有 column
all_features_from_encode = encode_step.get_feature_names_out()

# 3. 取得「篩選器」留下來的索引 (True/False)
# 這是一個布林遮罩，長度會等於 all_features_from_encode
is_selected = selector_step.get_support()

# 4. 根據遮罩，過濾出最後進入 Random Forest 的名稱
final_feature_names = all_features_from_encode[is_selected]

# 5. 建立 DataFrame (這時候 row 的數量就會完全對齊了)
importance = pd.DataFrame({
    'feature': final_feature_names, 
    'importance': model_step.feature_importances_
})

print("\n前 10 大重要特徵:")
print(importance.sort_values(by='importance', ascending=False).head(10))

# 根據模式決定儲存路徑
model_filename = "test_pipeline.joblib" if is_test else "best_pipeline.joblib"

# 儲存模型
joblib.dump(pipe, model_filename)
print(f"\n✅ 模型已成功儲存為 {model_filename}")


# train.py 的最後面
import joblib
print("嘗試重新載入模型...")
test_load = joblib.load(model_filename)
print("載入成功！代表環境沒問題。")

