import unittest
import pandas as pd
import numpy as np
from feature_builder import FeatureBuilder

class TestFeatureBuilder(unittest.TestCase):
    def setUp(self):
        """在每個測試開始前，先初始化 FeatureBuilder 實例"""
        self.builder = FeatureBuilder()

    def test_transform_columns(self):
        """測試轉換後的欄位(column)數量與名稱是否正確"""
        # 1. 準備模擬資料 (包含各種原始欄位)
        input_df = pd.DataFrame({
            'SBA_Appv': ['$50,000'],          # 測試金額清理
            'Bank': [np.nan],                 # 測試缺失值處理
            'NAICS': ['236115'],              # 測試取前兩碼
            'ApprovalDate': ['2006-01-01'],   # 測試日期轉換
            'FranchiseCode':['0'],             # 測試加盟代碼轉換
            'RevLineCr': [np.nan]             # 測試缺失值填充
        })

        # 2. 執行轉換
        output_df = self.builder.transform(input_df)

        # 3. 驗證新欄位是否存在
        self.assertIn('NAICS_Section', output_df.columns)
        self.assertIn('Days_Since_Appv', output_df.columns)
        self.assertIn('FranchiseCode_Binary', output_df.columns)
        self.assertIn('Is_Bank_Missing', output_df.columns)

        # 4. 驗證舊欄位是否已刪除
        self.assertNotIn('NAICS', output_df.columns)
        self.assertNotIn('ApprovalDate', output_df.columns)

    def test_currency_cleaning(self):
        """測試金額欄位是否成功轉為數字"""
        input_df = pd.DataFrame({'SBA_Appv': ['$1,234.56']})
        
        # 這裡會跑你 Notebook 裡面的那個 for c in cn 迴圈邏輯
        output_df = self.builder.transform(input_df)
        
        # 必須加上 才能抓到第一 row 的 1234.56
        self.assertEqual(output_df['SBA_Appv'].iloc[0], 1234.56)

if __name__ == '__main__':
    unittest.main()