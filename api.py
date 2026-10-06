"""
供應商文件判讀 API (給「供應商管理 App」呼叫)
------------------------------------------------
跟 Streamlit 介面共用同一套判讀核心(extract_core.py)，修正一次兩邊同時生效。

端點：
  GET  /health                 健康檢查
  POST /api/extract_vendor     上傳文件判讀(主 App 目前呼叫的網址)
  POST /api/extract            同上(新名稱，之後三種文件都用這個)

POST 參數(multipart/form-data)：
  file             PDF 或圖片
  document_type    supplier_form(供應商資料表，預設) | supplier_evaluation(供應商評核表) | quotation(報價單，尚未支援)
  template_version 目前只有 V1

環境變數(Cloud Run 上設定)：
  VENDOR_API_KEY             有設定的話，呼叫時必須帶 X-API-Key 標頭
  ALLOWED_ORIGINS            允許呼叫的網頁來源，逗號分隔；預設 * (全部允許)
  GCP_SERVICE_ACCOUNT_JSON   (選填) 服務帳戶金鑰 JSON 內容；不設定就用 Cloud Run 本身的服務帳戶
------------------------------------------------
"""

import json
import os
import re

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from extract_core import check_eval_record, file_to_images, load_template, run_extraction

API_VERSION = "2026-10-06-api-v1"
MAX_BYTES = 20 * 1024 * 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOC_TYPES = {
    "supplier_form": {"template": "templates/supplier_form.json", "pages": 1, "label": "供應商資料表"},
    "supplier_evaluation": {"template": "templates/eval_form.json", "pages": 2, "label": "供應商評核表"},
}
DOC_ALIASES = {"vendor_form": "supplier_form", "supplier": "supplier_form",
               "evaluation": "supplier_evaluation", "eval_form": "supplier_evaluation",
               "supplier_eval": "supplier_evaluation"}

# 供應商資料表：模板欄位名稱 -> 主 App 使用的英文 key
SUPPLIER_KEYS = {
    "公司全名(中文)": "company_name_zh",
    "公司全名(英文)": "company_name_en",
    "統一編號/身份證號": "tax_id",
    "服務範疇": "service_scope",
    "核准設立日期(年/月/日)": "established_date",
    "資本額(單位:萬元)": "capital",
    "聯絡地址": "contact_address",
    "帳單地址": "billing_address",
    "負責人": "responsible_person",
    "公司電話": "company_phone",
    "聯絡人": "contact_person",
    "連絡電話": "contact_phone",
    "匯款帳戶戶名": "bank_account_name",
    "匯款銀行": "bank_name",
    "分行別": "branch_name",
    "匯款帳號": "bank_account",
    "發票類型": "invoice_type",
    "課稅別": "tax_type",
    "填表人": "preparer",
    "填表日期": "form_date",
}

app = FastAPI(title="供應商文件判讀 API", version=API_VERSION)
origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"])

_vision_client = None


def get_vision_client():
    global _vision_client
    if _vision_client is None:
        from google.cloud import vision
        sa_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
        if sa_json:
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_info(json.loads(sa_json))
            _vision_client = vision.ImageAnnotatorClient(credentials=creds)
        else:
            _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


def check_key(x_api_key):
    expected = os.environ.get("VENDOR_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="API Key 錯誤或未提供(X-API-Key)")


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def build_supplier(values, template):
    normalized = {key: values.get(name, "") for name, key in SUPPLIER_KEYS.items()}
    normalized["attachments"] = {f["name"]: values.get(f["name"], "")
                                 for f in template["fields"] if f["name"].startswith("附件")}
    warns = []
    for f in template["fields"]:
        if f["type"] == "text" and f.get("required", True) and not str(values.get(f["name"], "")).strip():
            warns.append(f"「{f['name']}」沒有辨識到內容")
    tax = re.sub(r"\s", "", str(values.get("統一編號/身份證號", "")))
    if tax.isdigit() and len(tax) != 8:
        warns.append(f"統一編號應為 8 碼，辨識結果為「{tax}」")
    return normalized, warns


def build_evaluation(values):
    items = {}
    for it in ["a1", "a2", "b1", "b2", "c1", "c2", "c3", "d1", "d2", "d3"]:
        items[it] = {"max_score": values.get(f"{it}_評分", ""),   # 評分 = 這一項最高給分
                     "score": values.get(f"{it}_得分", ""),       # 得分 = 實際評核分數
                     "note": values.get(f"{it}_說明", "")}
    bonus = {}
    for it in ["e1", "e2"]:
        bonus[it] = {"meets": values.get(f"{it}_是否符合", ""),
                     "note": values.get(f"{it}_說明", ""),
                     "bonus": values.get(f"{it}_加分", "")}
    warns, subtotal, full = check_eval_record(values)
    normalized = {
        "supplier_name": values.get("供應商名稱", ""),
        "evaluation_category": values.get("評核類別", ""),
        "purchase_item": values.get("採購項目", ""),
        "evaluation_unit": values.get("評核單位", ""),
        "evaluator": values.get("評核人員", ""),
        "evaluation_date": values.get("評核日期", ""),
        "items": items,
        "bonus": bonus,
        "total_score_written": values.get("評核總分", ""),
        "grade_written": values.get("評核等級", ""),
        "unit_subtotal": subtotal,          # 本單位 得分合計 + 加分(加分單項上限 10)
        "unit_full_score": full,            # 本單位有給分項目的「最高給分」合計
    }
    return normalized, warns


@app.get("/health")
def health():
    return {"status": "ok", "version": API_VERSION, "document_types": list(DOC_TYPES) + ["quotation(尚未支援)"]}


@app.post("/api/extract_vendor")
@app.post("/api/extract")
async def extract(file: UploadFile = File(...),
                  document_type: str = Form("supplier_form"),
                  template_version: str = Form("V1"),
                  x_api_key: str = Header(None)):
    check_key(x_api_key)
    doc_type = DOC_ALIASES.get(document_type, document_type)
    if doc_type == "quotation":
        raise HTTPException(status_code=501, detail="報價單判讀尚未上線")
    if doc_type not in DOC_TYPES:
        raise HTTPException(status_code=400, detail=f"不支援的 document_type：{document_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="檔案是空的")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="檔案超過 20MB")

    cfg = DOC_TYPES[doc_type]
    template = load_template(os.path.join(BASE_DIR, cfg["template"]))
    warnings = []
    try:
        pages = file_to_images(data, file.filename, template)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取檔案：{e}")
    if len(pages) < cfg["pages"]:
        raise HTTPException(status_code=400,
                            detail=f"{cfg['label']}需要 {cfg['pages']} 頁，這份檔案只有 {len(pages)} 頁")
    if len(pages) > cfg["pages"]:
        warnings.append(f"檔案共 {len(pages)} 頁，只判讀前 {cfg['pages']} 頁"
                        + ("；多個單位的評核請分開上傳" if doc_type == "supplier_evaluation" else ""))

    try:
        values, _crops, _boxes = run_extraction(pages[:cfg["pages"]], template, get_vision_client(), warnings.append)
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"判讀失敗：{e}",
                                                      "api_version": API_VERSION})

    if doc_type == "supplier_form":
        normalized, more = build_supplier(values, template)
    else:
        normalized, more = build_evaluation(values)
    warnings.extend(more)

    return {
        "success": True,
        "api_version": API_VERSION,
        "document_type": doc_type,
        "template_version": template_version,
        "pages": len(pages),
        "normalized": normalized,
        "fields": values,                       # 模板原始欄位名稱 -> 值(除錯/對照用)
        "warnings": warnings,
    }
