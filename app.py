import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 設定頁面標題
st.set_page_config(page_title="SBA 貸款違約預測系統", layout="centered")

# 1. 載入模型 (快取處理，避免重複載入)
@st.cache_resource
def load_model():
    return joblib.load("best_pipeline.joblib")

model = load_model()

st.title("SBA 貸款信用風險評估")
st.markdown("請輸入貸款申請資訊，系統將自動評估違約機率。")

# 2. 建立輸入表單
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sba_appv = st.text_input("SBA 核准金額 (例如: $123,456)", "$50,000")
        state = st.selectbox("申請人所在州別", ["CA", "NY", "TX", "FL", "IL", "Other"])
        naics = st.text_input("NAICS 行業代碼 (前兩碼)", "33")
        
    with col2:
        term = st.number_input("貸款期限 (月)", min_value=1, max_value=360, value=60)
        new_exist = st.selectbox("公司類型", options=[1, 2], format_func=lambda x: "現有公司" if x==1 else "新創公司")
        urban_rural = st.selectbox("地區屬性", options=[0, 1, 2], format_func=lambda x: "未知" if x==0 else ("都市" if x==1 else "鄉村"))

    submitted = st.form_submit_with_button("開始分析風險")

# 3. 執行預測
if submitted:
    # 將輸入資料包裝成 DataFrame (欄位名稱必須與訓練時完全一致)
    input_data = pd.DataFrame({
        'SBA_Appv': [sba_appv],
        'State': [state],
        'NAICS': [naics],
        'Term': [term],
        'NewExist': [new_exist],
        'UrbanRural': [urban_rural],
        # 如果模型還有其他必要欄位，請在這裡補齊（可給預設值）
        'BankState': [state], 
        'ApprovalFY': [2026],
        'RevLineCr': ['N'],
        'LowDoc': ['N']
    })

    # 預測機率
    pd_score = model.predict_proba(input_data)[0, 1]
    
    # 顯示結果
    st.divider()
    st.subheader("分析結果")
    
    if pd_score < 0.2:
        st.success(f"低風險：違約機率 {pd_score:.2%}")
    elif pd_score < 0.5:
        st.warning(f"中等風險：違約機率 {pd_score:.2%}")
    else:
        st.error(f"高風險：違約機率 {pd_score:.2%}")

    # 視覺化儀表板 (簡單的進度條)
    st.progress(pd_score)