"""模組一：供應商資料主檔 ← 供應商資料表(1 頁)"""

import re

from fastapi import APIRouter, File, Form, Header, UploadFile

from .common import check_key, extract_form, ok_response, read_upload

MODULE = "supplier_master"
DOCUMENT_TYPE = "supplier_form"
TEMPLATE = "templates/supplier_form.json"
PAGES = 1

# 模板欄位名稱 -> 主 App 使用的英文 key
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

router = APIRouter(tags=["供應商資料主檔"])


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


def extract(data: bytes, filename: str, template_version: str = "V1"):
    values, template, pages, warnings = extract_form(data, filename, TEMPLATE, PAGES, "供應商資料表")
    normalized, more = build_supplier(values, template)
    return ok_response(MODULE, DOCUMENT_TYPE, pages, normalized, values, warnings + more,
                       template_version=template_version)


@router.post("/api/supplier-master/extract", summary="供應商資料表判讀")
async def extract_endpoint(file: UploadFile = File(..., description="供應商資料表 PDF 或圖片(1 頁)"),
                           template_version: str = Form("V1"),
                           x_api_key: str = Header(None)):
    check_key(x_api_key)
    return extract(await read_upload(file), file.filename, template_version)
