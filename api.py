"""
供應商文件判讀 API (給「供應商管理 App」呼叫)
------------------------------------------------
三個模組各一支判讀程式(modules/)，跟 Streamlit 介面共用同一套判讀核心(extract_core.py)，修正一次兩邊同時生效。

端點：
  GET  /health                              健康檢查(列出這個服務有開的模組)
  POST /api/supplier-master/extract         模組一 供應商資料主檔：供應商資料表
  POST /api/supplier-evaluation/extract     模組二 供應商評核：供應商評核表
  POST /api/quotation/extract               模組三 日常報價比價：報價單(第一版：表頭與金額，品項明細待樣本)
  POST /api/extract、/api/extract_vendor    舊網址，用 document_type 分派到上面三個模組(相容用)
  GET  /docs、/openapi.json                 自動產生的 API 文件 / OpenAPI 規格

POST 參數(multipart/form-data)：
  file             PDF 或圖片
  template_version 目前只有 V1
  document_type    只有舊網址需要：supplier_form | supplier_evaluation | quotation

環境變數(Cloud Run 上設定)：
  VENDOR_API_KEY             有設定的話，呼叫時必須帶 X-API-Key 標頭
  ALLOWED_ORIGINS            允許呼叫的網頁來源，逗號分隔；預設 * (全部允許)
  GCP_SERVICE_ACCOUNT_JSON   (選填) 服務帳戶金鑰 JSON 內容；不設定就用 Cloud Run 本身的服務帳戶
  ENABLED_MODULES            (選填) 這個服務要開哪些模組，逗號分隔；預設三個全開。
                             想拆成三個獨立服務時，每個服務各設一個，例如 supplier_master
------------------------------------------------
"""

import os

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from modules import quotation, supplier_evaluation, supplier_master
from modules.common import API_VERSION, check_key, read_upload

ALL_MODULES = {m.MODULE: m for m in (supplier_master, supplier_evaluation, quotation)}
enabled = [m.strip() for m in os.environ.get("ENABLED_MODULES", ",".join(ALL_MODULES)).split(",") if m.strip()]
unknown = [m for m in enabled if m not in ALL_MODULES]
if unknown:
    raise RuntimeError(f"ENABLED_MODULES 有不認得的模組：{unknown}，可用：{list(ALL_MODULES)}")
MODULES = {name: ALL_MODULES[name] for name in enabled}

# 舊網址 document_type -> 模組
DOC_TYPE_TO_MODULE = {
    "supplier_form": "supplier_master", "vendor_form": "supplier_master", "supplier": "supplier_master",
    "supplier_evaluation": "supplier_evaluation", "evaluation": "supplier_evaluation",
    "eval_form": "supplier_evaluation", "supplier_eval": "supplier_evaluation",
    "quotation": "quotation", "quote": "quotation",
}

app = FastAPI(title="供應商文件判讀 API", version=API_VERSION)
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"])
for _m in MODULES.values():
    app.include_router(_m.router)


@app.exception_handler(HTTPException)
async def http_error(_request, exc: HTTPException):
    # 錯誤也回統一格式；保留 detail 讓舊版 App 照樣讀得到
    return JSONResponse(status_code=exc.status_code,
                        content={"success": False, "api_version": API_VERSION,
                                 "message": exc.detail, "detail": exc.detail})


@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION, "modules": list(MODULES)}


@app.post("/api/extract_vendor", include_in_schema=False)
@app.post("/api/extract", summary="(舊網址)依 document_type 分派到對應模組")
async def extract(file: UploadFile = File(...),
                  document_type: str = Form("supplier_form"),
                  template_version: str = Form("V1"),
                  x_api_key: str = Header(None)):
    check_key(x_api_key)
    module_name = DOC_TYPE_TO_MODULE.get(document_type)
    if module_name is None:
        raise HTTPException(status_code=400, detail=f"不支援的 document_type：{document_type}")
    if module_name not in MODULES:
        raise HTTPException(status_code=404, detail=f"這個服務沒有開啟 {module_name} 模組")
    return MODULES[module_name].extract(await read_upload(file), file.filename, template_version)
