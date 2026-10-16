"""模組二：供應商評核 ← 供應商評核表(2 頁 = 一個單位的評核；新舊版面自動判斷)"""

from fastapi import APIRouter, File, Form, Header, UploadFile

from extract_core import check_eval_record

from .common import check_key, extract_form, ok_response, read_upload

MODULE = "supplier_evaluation"
DOCUMENT_TYPE = "supplier_evaluation"
TEMPLATE = "templates/eval_form.json"
PAGES = 2
ITEMS = ["a1", "a2", "b1", "b2", "c1", "c2", "c3", "d1", "d2", "d3"]
BONUS_ITEMS = ["e1", "e2"]

router = APIRouter(tags=["供應商評核"])


def build_evaluation(values):
    items = {}
    for it in ITEMS:
        items[it] = {"max_score": values.get(f"{it}_評分", ""),   # 評分 = 這一項最高給分
                     "score": values.get(f"{it}_得分", ""),       # 得分 = 實際評核分數
                     "note": values.get(f"{it}_說明", "")}
    bonus = {}
    for it in BONUS_ITEMS:
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


def extract(data: bytes, filename: str, template_version: str = "V1"):
    values, _template, pages, warnings = extract_form(data, filename, TEMPLATE, PAGES, "供應商評核表",
                                                      "；多個單位的評核請分開上傳")
    normalized, more = build_evaluation(values)
    return ok_response(MODULE, DOCUMENT_TYPE, pages, normalized, values, warnings + more,
                       template_version=template_version)


@router.post("/api/supplier-evaluation/extract", summary="供應商評核表判讀")
async def extract_endpoint(file: UploadFile = File(..., description="供應商評核表 PDF 或圖片(2 頁，一個單位)"),
                           template_version: str = Form("V1"),
                           x_api_key: str = Header(None)):
    check_key(x_api_key)
    return extract(await read_upload(file), file.filename, template_version)
