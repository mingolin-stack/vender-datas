"""
模組三：日常報價比價 ← 供應商報價單

報價單每家供應商格式都不同，不能用固定座標模板，改用「整頁文字 + 關鍵字規則」：
  - 電子檔 PDF(有內嵌文字)：直接讀出文字，不用呼叫 Vision
  - 掃描檔 PDF / 圖片：每頁送 Vision document_text_detection 取得全文
再從全文找出表頭資訊與金額(供應商、統編、日期、單號、有效期限、小計、稅額、總計)。

第一版還不拆品項明細(items 先回空陣列)，等拿到實際報價單樣本後再補上；
raw_text 會一併回傳，App 端或人工可以先對照。
"""

import re

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from extract_core import IMAGE_EXTS, RENDER_DPI

from . import common
from .common import check_key, ok_response, read_upload

MODULE = "quotation"
DOCUMENT_TYPE = "quotation"
MAX_PAGES = 10

router = APIRouter(tags=["日常報價比價"])

_NUM = r"(?:NT\$|\$|TWD)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)"
_DATE = r"((?:19|20)\d{2}|1\d{2})\s*[./\-年]\s*(\d{1,2})\s*[./\-月]\s*(\d{1,2})\s*日?"
# 這些開頭的行是「客戶/收件方」，裡面的公司名稱不是報價的供應商
_BUYER_PREFIX = re.compile(r"^\s*(客戶|客户|致|TO|To|收件|買方|買受人|訂購|貴公司|敬致|Attn)", re.I)


def pdf_text_pages(data: bytes):
    """電子檔 PDF 每頁的內嵌文字；掃描檔通常是空字串。"""
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    return [page.get_text() for page in doc]


def ocr_page(vision_client, pil_image) -> str:
    from google.cloud import vision
    from vision_utils import image_to_bytes
    response = vision_client.document_text_detection(image=vision.Image(content=image_to_bytes(pil_image)),
                                                     image_context={"language_hints": ["zh-TW", "zh", "en"]})
    if response.error.message:
        raise RuntimeError(f"Vision API 錯誤: {response.error.message}")
    return response.full_text_annotation.text if response.full_text_annotation else ""


def file_to_text(data: bytes, filename: str, warnings: list):
    """回傳 (全文, 頁數, 文字來源)。"""
    from extract_core import file_to_images, pdf_to_images
    name = (filename or "").lower()
    is_pdf = name.endswith(".pdf") or data[:4] == b"%PDF"
    if not is_pdf and not name.endswith(IMAGE_EXTS):
        raise HTTPException(status_code=400, detail=f"不支援的檔案格式：{filename}（請上傳 PDF 或圖片）")
    try:
        if is_pdf:
            texts = pdf_text_pages(data)
            if texts and all(len(t.strip()) >= 20 for t in texts):
                return "\n".join(texts[:MAX_PAGES]), len(texts), "pdf_text"
            pages = pdf_to_images(data, dpi=RENDER_DPI)
        else:
            pages = file_to_images(data, filename)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法讀取檔案：{e}")
    if len(pages) > MAX_PAGES:
        warnings.append(f"檔案共 {len(pages)} 頁，只判讀前 {MAX_PAGES} 頁")
    try:
        client = common.get_vision_client()
        text = "\n".join(ocr_page(client, p) for p in pages[:MAX_PAGES])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"判讀失敗：{e}")
    return text, len(pages), "ocr"


def _to_number(s):
    try:
        v = float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    return int(v) if v.is_integer() else v


def _find_amount(lines, keywords, exclude=()):
    """找含關鍵字那一行(或下一行)的最後一個數字；有多行符合時取最後出現的(總計通常在最下面)。"""
    found = None
    for i, line in enumerate(lines):
        if any(k in line for k in keywords) and not any(x in line for x in exclude):
            after = line.split(next(k for k in keywords if k in line), 1)[1]
            nums = re.findall(_NUM, after) or (re.findall(_NUM, lines[i + 1]) if i + 1 < len(lines) else [])
            nums = [n for n in nums if _to_number(n) is not None]
            if nums:
                found = _to_number(nums[-1])
    return found


def _find_date(lines, keywords, exclude=()):
    for i, line in enumerate(lines):
        if any(k in line for k in keywords) and not any(x in line for x in exclude):
            m = re.search(_DATE, line) or (re.search(_DATE, lines[i + 1]) if i + 1 < len(lines) else None)
            if m:
                y, mo, d = (int(g) for g in m.groups())
                if y < 1911:
                    y += 1911       # 民國年
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    return f"{y:04d}-{mo:02d}-{d:02d}"
    return ""


def _find_after(lines, keywords, pattern=r"[A-Za-z0-9\-_/]+"):
    for i, line in enumerate(lines):
        for k in keywords:
            if k in line:
                rest = line.split(k, 1)[1].lstrip(" :：#.")
                m = re.match(pattern, rest)
                if not m and i + 1 < len(lines):
                    m = re.match(pattern, lines[i + 1].strip())
                if m:
                    return m.group(0)
    return ""


def parse_quotation(text: str):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    warns = []

    supplier = ""
    for line in lines:
        if _BUYER_PREFIX.match(line):
            continue
        m = re.search(r"([一-鿿A-Za-z0-9（）()&\s]{2,30}?(?:股份有限公司|有限公司|企業社|工程行|商行|行號|公司))", line)
        if m:
            supplier = m.group(1).strip()
            break

    tax_id = _find_after(lines, ["統一編號", "統編", "營利事業統一編號"], r"\d{8}")
    if not tax_id:
        ids = re.findall(r"(?<!\d)\d{8}(?!\d)", text)
        tax_id = ids[0] if len(ids) == 1 else ""

    subtotal = _find_amount(lines, ["未稅合計", "未稅金額", "未稅總價", "小計", "合計(未稅)"])
    tax = _find_amount(lines, ["營業稅", "稅額", "稅金", "5%稅"])
    total = _find_amount(lines, ["含稅總計", "含稅合計", "含稅總價", "總計", "總金額", "總價", "合計金額", "TOTAL", "Total"],
                         exclude=["未稅", "小計"])
    if total is None and subtotal is not None and tax is not None:
        total = subtotal + tax
    if subtotal is not None and tax is not None and total is not None and abs(subtotal + tax - total) > 1:
        warns.append(f"小計 {subtotal} + 稅額 {tax} 不等於總計 {total}，請核對")

    tax_included = None
    if "含稅" in text:
        tax_included = True
    elif "未稅" in text or "稅外加" in text:
        tax_included = False

    normalized = {
        "supplier_name": supplier,
        "tax_id": tax_id,
        "quote_no": _find_after(lines, ["報價單號", "報價編號", "估價單號", "單號", "Quotation No", "NO."]),
        "quote_date": _find_date(lines, ["報價日期", "估價日期", "日期", "Date"],
                                 exclude=["有效", "交貨", "交期", "Valid"]),
        "valid_until": _find_date(lines, ["有效期限", "有效日期", "報價有效", "Valid"]),
        "currency": "USD" if re.search(r"\bUSD\b|US\$", text) else "TWD",
        "tax_included": tax_included,
        "items": [],                 # 品項明細：等報價單樣本後再支援
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }
    for key, label in [("supplier_name", "供應商名稱"), ("quote_date", "報價日期"), ("total", "總計金額")]:
        if normalized[key] in ("", None):
            warns.append(f"沒有找到{label}")
    warns.append("品項明細尚未支援，items 先回空陣列；請以 raw_text 對照")
    return normalized, warns


def extract(data: bytes, filename: str, template_version: str = "V1"):
    warnings = []
    text, pages, source = file_to_text(data, filename, warnings)
    if not text.strip():
        raise HTTPException(status_code=400, detail="這份檔案讀不到任何文字")
    normalized, more = parse_quotation(text)
    return ok_response(MODULE, DOCUMENT_TYPE, pages, normalized, {}, warnings + more,
                       text_source=source, raw_text=text)


@router.post("/api/quotation/extract", summary="報價單判讀(第一版：表頭與金額)")
async def extract_endpoint(file: UploadFile = File(..., description="報價單 PDF 或圖片"),
                           x_api_key: str = Header(None)):
    check_key(x_api_key)
    return extract(await read_upload(file), file.filename)
