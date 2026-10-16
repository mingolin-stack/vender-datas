"""
三個模組各自一支判讀程式，對應「供應商管理 App」的三個區塊：

  supplier_master      供應商資料主檔  ← 供應商資料表
  supplier_evaluation  供應商評核      ← 供應商評核表
  quotation            日常報價比價    ← 報價單(待提供樣本後開發)

每個模組都是一個 FastAPI APIRouter，api.py 依環境變數 ENABLED_MODULES 決定要掛哪幾個，
所以可以三個模組放在同一個服務，也可以部署成三個獨立服務(同一份程式碼、不同設定)。
"""
