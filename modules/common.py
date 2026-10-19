"""三個模組共用：Vision 用戶端、API Key 檢查、上傳檔案讀取、回傳格式。"""

import json
import os

from fastapi import HTTPException, UploadFile

from extract_core import file_to_images, load_template, run_extraction

API_VERSION = "2026-10-16-api-v2"
MAX_BYTES = 20 * 1024 * 1024
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="檔案是空的")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="檔案超過 20MB")
    return data


def extract_form(data: bytes, filename: str, template_rel: str, pages_needed: int, label: str, split_hint: str = ""):
    """模板式表單共用流程：轉圖 → 檢查頁數 → 跑判讀。回傳 (欄位值, 模板, 總頁數, 警告)。"""
    template = load_template(os.path.join(BASE_DIR, template_rel))
    warnings = []
    try:
        pages = file_to_images(data, filename, template)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取檔案：{e}")
    if len(pages) < pages_needed:
        raise HTTPException(status_code=400, detail=f"{label}需要 {pages_needed} 頁，這份檔案只有 {len(pages)} 頁")
    if len(pages) > pages_needed:
        warnings.append(f"檔案共 {len(pages)} 頁，只判讀前 {pages_needed} 頁{split_hint}")
    try:
        values, _crops, _boxes = run_extraction(pages[:pages_needed], template, get_vision_client(), warnings.append)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"判讀失敗：{e}")
    return values, template, len(pages), warnings


def ok_response(module: str, document_type: str, pages: int, normalized: dict, fields: dict, warnings: list,
                **extra):
    """三個模組統一的成功回傳格式。"""
    return {
        "success": True,
        "api_version": API_VERSION,
        "module": module,
        "document_type": document_type,
        "pages": pages,
        "normalized": normalized,   # 給 App 直接用的英文 key
        "fields": fields,           # 模板原始欄位名稱 -> 值(除錯/對照用)
        "warnings": warnings,
        **extra,
    }
