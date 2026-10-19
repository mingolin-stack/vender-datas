"""
模組三：日常報價比價 ← 供應商報價單

報價單每家供應商格式都不同，不能用固定座標模板，改用「整頁文字 + 關鍵字規則」：
  - 電子檔 PDF(有內嵌文字)：直接讀出文字，不用呼叫 Vision
  - 掃描檔 PDF / 圖片：每頁送 Vision document_text_detection 取得全文
  - Excel(.xlsx)：直接讀儲存格
再從全文找出表頭資訊與金額(供應商、統編、日期、單號、有效期限、小計、稅額、總計)。

品項明細用「數量 × 單價 = 金額」找出每一列；Excel 報價單直接讀儲存格。
規則是用自製樣本寫的，拿到實際報價單後要再調整；raw_text 會一併回傳，App 端或人工可以對照。
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


def rows_from_words(words):
    """
    把 (x0, y0, x1, y1, 文字) 依高度分成一列一列，再由左到右接起來。
    報價單的品項表格，OCR 或 PDF 原始順序常常是一欄一欄讀，這樣才能還原成「一列一個品項」。
    """
    words = [w for w in words if str(w[4]).strip()]
    if not words:
        return ""
    heights = sorted(w[3] - w[1] for w in words)
    tol = max(heights[len(heights) // 2] * 0.5, 1)
    rows = []
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        yc = (w[1] + w[3]) / 2
        if rows and abs(yc - rows[-1]["yc"]) <= tol:
            rows[-1]["words"].append(w)
            n = len(rows[-1]["words"])
            rows[-1]["yc"] += (yc - rows[-1]["yc"]) / n
        else:
            rows.append({"yc": yc, "words": [w]})
    lines = []
    for row in rows:
        ws = sorted(row["words"], key=lambda w: w[0])
        line = str(ws[0][4])
        for prev, cur in zip(ws, ws[1:]):
            gap = cur[0] - prev[2]
            line += ("  " if gap > tol * 2 else "") + str(cur[4])
        lines.append(line)
    return "\n".join(lines)


def pdf_text_pages(data: bytes):
    """電子檔 PDF 每頁的內嵌文字(依位置排成列)；掃描檔通常是空字串。"""
    import fitz
    doc = fitz.open(stream=data, filetype="pdf")
    return [rows_from_words([w[:5] for w in page.get_text("words")]) for page in doc]


def ocr_page(vision_client, pil_image) -> str:
    from google.cloud import vision
    from vision_utils import image_to_bytes
    response = vision_client.document_text_detection(image=vision.Image(content=image_to_bytes(pil_image)),
                                                     image_context={"language_hints": ["zh-TW", "zh", "en"]})
    if response.error.message:
        raise RuntimeError(f"Vision API 錯誤: {response.error.message}")
    words = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    xs = [v.x for v in word.bounding_box.vertices]
                    ys = [v.y for v in word.bounding_box.vertices]
                    words.append((min(xs), min(ys), max(xs), max(ys), "".join(sym.text for sym in word.symbols)))
    return rows_from_words(words)


def excel_text(data: bytes):
    """Excel 報價單：每個工作表的每一列，儲存格之間用兩個空白隔開。"""
    from io import BytesIO
    from openpyxl import load_workbook
    wb = load_workbook(BytesIO(data), data_only=True, read_only=True)
    lines = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [_cell_str(c) for c in row if c is not None and str(c).strip()]
            if cells:
                lines.append("  ".join(cells))
    return "\n".join(lines), len(wb.worksheets)


def _cell_str(c):
    if hasattr(c, "strftime"):
        return c.strftime("%Y/%m/%d")
    if isinstance(c, float) and c.is_integer():
        return str(int(c))
    return str(c).strip()


def file_to_text(data: bytes, filename: str, warnings: list):
    """回傳 (全文, 頁數, 文字來源)。"""
    from extract_core import file_to_images, pdf_to_images
    name = (filename or "").lower()
    is_pdf = name.endswith(".pdf") or data[:4] == b"%PDF"
    is_excel = name.endswith((".xlsx", ".xlsm"))
    if not (is_pdf or is_excel or name.endswith(IMAGE_EXTS)):
        raise HTTPException(status_code=400, detail=f"不支援的檔案格式：{filename}（請上傳 PDF、Excel 或圖片）")
    try:
        if is_excel:
            text, sheets = excel_text(data)
            return text, sheets, "excel"
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
    """找關鍵字後面的第一個數字(同一行沒有就看下一行)；有多行符合時取最後出現的(總計通常在最下面)。"""
    found = None
    for i, line in enumerate(lines):
        if any(k in line for k in keywords) and not any(x in line for x in exclude):
            after = line.split(next(k for k in keywords if k in line), 1)[1]
            nums = re.findall(_NUM, after) or (re.findall(_NUM, lines[i + 1]) if i + 1 < len(lines) else [])
            nums = [n for n in nums if _to_number(n) is not None]
            if nums:
                found = _to_number(nums[0])
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


_UNITS = ("公斤", "公噸", "公尺", "平方公尺", "立方公尺", "坪", "才", "式", "組", "套", "個", "支", "片", "台", "臺",
          "箱", "包", "件", "張", "桶", "卷", "捲", "本", "盒", "顆", "塊", "條", "根", "瓶", "罐", "袋", "只", "天", "次",
          "人", "批", "kg", "KG", "pcs", "PCS", "set", "SET", "EA", "ea", "m", "M")
_TOTAL_WORDS = ("合計", "小計", "總計", "總價", "總金額", "稅", "TOTAL", "Total", "折扣", "優惠")
_ITEM_NUM = re.compile(r"(?<![\w.])\$?([0-9][0-9,]*(?:\.[0-9]+)?)(?![\w.])")


def parse_items(lines):
    """
    品項明細：一列裡面找三個數字「數量 × 單價 = 金額」(誤差 1 元內)，前面的文字當品名。
    格式各家不同，找不到就回空陣列，並在 warnings 提醒。
    """
    items = []
    for line in lines:
        if any(w in line for w in _TOTAL_WORDS):
            continue
        matches = list(_ITEM_NUM.finditer(line))
        if len(matches) < 3:
            continue
        vals = [_to_number(m.group(1)) for m in matches]
        hit = None
        # 從右邊找：金額通常在最後，單價、數量在它左邊(數量和單價誰先都可以)
        for k in range(len(vals) - 1, 1, -1):
            for j in range(k - 1, 0, -1):
                for i in range(j - 1, -1, -1):
                    a, b, c = vals[i], vals[j], vals[k]
                    if None in (a, b, c) or c == 0:
                        continue
                    if abs(a * b - c) <= max(1, c * 0.001):
                        hit = (i, j, k)
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            continue
        i, j, k = hit
        name_part = line[:matches[i].start()].strip()
        name_part = re.sub(r"^\d{1,3}[.、)\s]+", "", name_part)      # 去掉項次編號
        between = line[matches[i].end():matches[k].start()]
        unit = next((u for u in _UNITS if u in between), "") or next((u for u in _UNITS if name_part.endswith(u)), "")
        if unit and name_part.endswith(unit):
            name_part = name_part[: -len(unit)].strip()
        if not re.search(r"[\u4e00-\u9fffA-Za-z]", name_part):
            continue
        parts = re.split(r"\s{2,}", name_part, maxsplit=1)
        qty, price = vals[i], vals[j]     # 台灣報價單多半是「數量 單價 金額」的順序
        items.append({"name": parts[0], "spec": parts[1] if len(parts) > 1 else "", "unit": unit,
                      "quantity": qty, "unit_price": price, "amount": vals[k],
                      "note": line[matches[k].end():].strip()})
    return items


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
        "items": parse_items(lines),
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }
    for key, label in [("supplier_name", "供應商名稱"), ("quote_date", "報價日期"), ("total", "總計金額")]:
        if normalized[key] in ("", None):
            warns.append(f"沒有找到{label}")
    items = normalized["items"]
    if not items:
        warns.append("沒有找到品項明細(數量 × 單價 = 金額)，請以 raw_text 對照")
    elif subtotal is not None:
        s_items = sum(it["amount"] for it in items)
        if abs(s_items - subtotal) > 1:
            warns.append(f"品項金額合計 {s_items} 跟小計 {subtotal} 不同，可能有品項沒讀到，請核對")
    return normalized, warns


def extract(data: bytes, filename: str, template_version: str = "V1"):
    warnings = []
    text, pages, source = file_to_text(data, filename, warnings)
    if not text.strip():
        raise HTTPException(status_code=400, detail="這份檔案讀不到任何文字")
    normalized, more = parse_quotation(text)
    return ok_response(MODULE, DOCUMENT_TYPE, pages, normalized, {}, warnings + more,
                       text_source=source, raw_text=text)


@router.post("/api/quotation/extract", summary="報價單判讀")
async def extract_endpoint(file: UploadFile = File(..., description="報價單 PDF、Excel 或圖片"),
                           x_api_key: str = Header(None)):
    check_key(x_api_key)
    return extract(await read_upload(file), file.filename)
