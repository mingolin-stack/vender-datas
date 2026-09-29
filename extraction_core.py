"""Reusable supplier-form extraction core shared by Streamlit UI and HTTP API."""
from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path
from typing import Callable

import fitz
from PIL import Image

from auto_align import compute_zone_transform, remap_box, compute_row_sequence, remap_box_row_sequence
from vision_utils import crop_field, ocr_text, is_checked

RENDER_DPI = 200
ROOT = Path(__file__).resolve().parent


def load_template(template_path: str | Path) -> dict:
    path = Path(template_path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pdf_to_images(pdf_bytes: bytes, dpi: int = RENDER_DPI) -> list[Image.Image]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return images


def image_bytes_to_images(data: bytes) -> list[Image.Image]:
    image = Image.open(BytesIO(data)).convert("RGB")
    return [image]


def document_to_images(data: bytes, filename: str = "", content_type: str = "") -> list[Image.Image]:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    if name.endswith(".pdf") or ctype == "application/pdf":
        return pdf_to_images(data)
    if any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")) or ctype.startswith("image/"):
        return image_bytes_to_images(data)
    raise ValueError("目前自動辨識支援 PDF、PNG、JPG/JPEG、WEBP、BMP、TIFF；Word 請先另存為 PDF。")


def get_field_page_image(page_images: list[Image.Image], field_or_option: dict) -> Image.Image:
    page_no = field_or_option.get("page", 1)
    idx = page_no - 1
    if idx >= len(page_images):
        raise ValueError(f"這份文件只有 {len(page_images)} 頁，但欄位設定要讀第 {page_no} 頁，頁數對不上。")
    return page_images[idx]


def resolve_box(field_or_option: dict, page_images: list[Image.Image], template: dict, zone_transform_cache: dict):
    box = field_or_option["box"]
    zone_name = field_or_option.get("zone")
    zones = template.get("zones")
    if not zone_name or not zones or zone_name not in zones:
        return box
    zone = zones[zone_name]
    page_no = field_or_option.get("page", 1)
    cache_key = (page_no, zone_name)
    if zone.get("type") == "rows":
        if cache_key not in zone_transform_cache:
            page_img = get_field_page_image(page_images, field_or_option)
            zone_transform_cache[cache_key] = compute_row_sequence(page_img, zone)
        return remap_box_row_sequence(box, zone_transform_cache[cache_key])
    if cache_key not in zone_transform_cache:
        page_img = get_field_page_image(page_images, field_or_option)
        zone_transform_cache[cache_key] = compute_zone_transform(page_img, zone)
    return remap_box(box, zone_transform_cache[cache_key])


def run_extraction(
    page_images: list[Image.Image],
    template: dict,
    vision_client,
    warning_callback: Callable[[str], None] | None = None,
) -> tuple[dict, dict]:
    suggestions: dict = {}
    crops: dict = {}
    zone_transform_cache: dict = {}
    for field in template["fields"]:
        if field["type"] == "text":
            img = get_field_page_image(page_images, field)
            box = resolve_box(field, page_images, template, zone_transform_cache)
            crop = crop_field(img, box)
            crops[field["name"]] = crop
            try:
                suggestions[field["name"]] = ocr_text(vision_client, crop)
            except Exception as exc:
                suggestions[field["name"]] = ""
                if warning_callback:
                    warning_callback(f"「{field['name']}」辨識失敗：{exc}")
        elif field["type"] == "checkbox_single":
            img = get_field_page_image(page_images, field)
            box = resolve_box(field, page_images, template, zone_transform_cache)
            crop = crop_field(img, box)
            crops[field["name"]] = crop
            checked, _ = is_checked(img, box)
            suggestions[field["name"]] = "是" if checked else "否"
        elif field["type"] == "checkbox_group":
            best_label, best_ratio = None, 0
            option_crops = []
            for opt in field["options"]:
                img = get_field_page_image(page_images, opt)
                box = resolve_box(opt, page_images, template, zone_transform_cache)
                crop = crop_field(img, box)
                option_crops.append((opt["label"], crop))
                checked, ratio = is_checked(img, box)
                if checked and ratio > best_ratio:
                    best_label, best_ratio = opt["label"], ratio
            crops[field["name"]] = option_crops
            suggestions[field["name"]] = best_label or ""
    return suggestions, crops


NORMALIZED_KEYS = {
    "公司全名(中文)": "company_name_zh",
    "公司全名(英文)": "company_name_en",
    "統一編號/身份證號": "tax_id",
    "服務範疇": "service_scope",
    "核准設立日期(年/月/日)": "established_date",
    "資本額(單位:萬元)": "capital_amount",
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


def normalize_supplier_fields(fields: dict) -> dict:
    return {alias: fields.get(label, "") for label, alias in NORMALIZED_KEYS.items()}


def _clean_space(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _clean_cjk_spacing(value: str) -> str:
    """Remove OCR-inserted spaces inside Chinese words while preserving Latin word spaces."""
    value = _clean_space(value)
    value = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"\s*([，,；;、])\s*", r"\1", value)
    return value.strip()


def _clean_person_name(value: str) -> str:
    value = _clean_space(value)
    compact = re.sub(r"\s+", "", value)
    if re.fullmatch(r"[\u4e00-\u9fff]{2,6}", compact):
        return compact
    return value


def _between(text: str, start_markers: tuple[str, ...], end_markers: tuple[str, ...]) -> str:
    """Return a conservative slice between form labels from flattened OCR text."""
    start_pos = -1
    start_len = 0
    for marker in start_markers:
        m = re.search(re.escape(marker), text, flags=re.I)
        if m and (start_pos < 0 or m.start() < start_pos):
            start_pos, start_len = m.start(), len(m.group(0))
    if start_pos < 0:
        return ""
    tail = text[start_pos + start_len :]
    end_pos = len(tail)
    for marker in end_markers:
        m = re.search(re.escape(marker), tail, flags=re.I)
        if m:
            end_pos = min(end_pos, m.start())
    return _clean_space(tail[:end_pos]).strip(" ：:#")


def postprocess_supplier_fields(
    fields: dict,
    full_text: str = "",
    filename: str = "",
) -> tuple[dict, list[str]]:
    """Clean common label bleed / crop-shift errors without inventing unsupported values.

    The first extraction remains the source of truth. Full-page OCR is used only when a
    field is clearly invalid or contaminated by form labels.
    """
    out = {str(k).strip(): _clean_space(v) if isinstance(v, str) else v for k, v in fields.items()}
    warnings: list[str] = []
    text = _clean_space(full_text)

    # Tax ID: Taiwanese company tax IDs are 8 digits. Prefer an 8-digit sequence found
    # in the tax/service header area, then fall back to the whole first page.
    tax_raw = out.get("統一編號/身份證號", "")
    tax_match = re.search(r"(?<!\d)(\d{8})(?!\d)", tax_raw)
    if not tax_match:
        header_area = _between(text, ("統一編號/身份證號", "統一編號", "身份證號"), ("核准設立日期", "服務範疇", "資本額", "聯絡地址"))
        tax_match = re.search(r"(?<!\d)(\d{8})(?!\d)", header_area) or re.search(r"(?<!\d)(\d{8})(?!\d)", text)
    if tax_match:
        out["統一編號/身份證號"] = tax_match.group(1)
    elif tax_raw and not re.fullmatch(r"\d{8}", re.sub(r"\D", "", tax_raw)):
        warnings.append("統一編號未通過 8 位數格式檢查，請人工確認。")

    # Company Chinese name: crop shift commonly places it in the English-name crop.
    zh = out.get("公司全名(中文)", "")
    en_raw = out.get("公司全名(英文)", "")
    if not zh:
        m = re.search(r"中文\s*[:：#]?\s*([^#]{2,60}?)(?=\s*(?:英文|統一編號|身份證號|\d{8}))", text, flags=re.I)
        if m:
            zh = _clean_space(m.group(1))
        else:
            m2 = re.match(r"中文\s*[:：#]?\s*(.+)", en_raw)
            if m2:
                zh = _clean_space(m2.group(1))
    zh = re.sub(r"^(中文|公司\s*全名(?:\s*\(中文\))?)\s*[:：#]?\s*", "", zh).strip()
    # OCR can append the neighbouring label after the actual company name.
    zh = re.sub(r"\s*(?:公司\s*全名(?:\s*\(中文\))?|中文)\s*$", "", zh).strip()
    zh = _clean_cjk_spacing(zh)
    if zh:
        out["公司全名(中文)"] = zh

    # English company name: derive from the header only when the crop is clearly the
    # Chinese-name value. Do not force a value if none is supported.
    en = en_raw
    if re.match(r"^中文\b|^中文\s", en) or (zh and zh in en):
        en = ""
    if not en:
        candidate = _between(text, ("英文#", "英文＃", "英文", "公司全名(英文)"), ("統一編號/身份證號", "統一編號", "服務範疇", "核准設立日期"))
        candidate = re.sub(r"(?<!\d)\d{8}(?!\d).*?$", "", candidate).strip()
        if candidate and not re.search(r"[\u4e00-\u9fff]{4,}", candidate):
            en = candidate
    out["公司全名(英文)"] = en

    # Service scope: strip form labels, names, and tax ID that leaked from neighboring rows.
    scope = out.get("服務範疇", "")
    scope = re.sub(r"^(英文[#＃]?|服務範疇)\s*[:：#]?\s*", "", scope, flags=re.I)
    for contaminant in (out.get("公司全名(英文)", ""), out.get("公司全名(中文)", ""), out.get("統一編號/身份證號", "")):
        if contaminant:
            scope = scope.replace(contaminant, " ")
    scope = re.sub(r"(?<!\d)\d{8}(?!\d)", " ", scope)
    out["服務範疇"] = _clean_cjk_spacing(scope).strip(" ,，;；")

    # Capital amount: keep explicit 萬元 values. If crop is just unrelated digits,
    # recover an amount near the 資本額 label from full-page OCR.
    capital = out.get("資本額(單位:萬元)", "")
    if not re.search(r"萬元", capital):
        cap_area = _between(text, ("資本額",), ("聯絡地址", "帳單地址", "負責人"))
        m = re.search(r"(\d+(?:\.\d+)?)\s*萬元(?:整)?", cap_area)
        if m:
            capital = m.group(0).replace(" ", "")
        elif capital and re.fullmatch(r"\d{5,}", re.sub(r"[,，]", "", capital)):
            warnings.append(f"資本額「{capital}」看起來可能是座標誤讀，請人工確認。")
    capital = re.sub(r"(?<=\d)\s+(?=萬)", "", capital)
    out["資本額(單位:萬元)"] = capital

    # Person names are often split by OCR with a space between Chinese characters.
    for key in ("負責人", "聯絡人"):
        out[key] = _clean_person_name(out.get(key, ""))

    # Address crops sometimes include the neighboring '(單位:萬元)' label.
    for key in ("聯絡地址", "帳單地址"):
        value = out.get(key, "")
        value = re.sub(r"\s*[（(]?單位\s*[:：]?\s*萬元[）)]?\s*$", "", value)
        out[key] = _clean_cjk_spacing(value)

    # Filename / preparer disagreement is useful as a warning, but is not safe enough
    # to overwrite OCR automatically.
    fname = Path(filename).stem if filename else ""
    preparer = out.get("填表人", "")
    # The preparer crop may drift upward into the attachment checklist. Attempt a
    # conservative full-page recovery; otherwise blank it and ask for confirmation.
    attachment_noise = ("供應商誠信廉潔", "保密承諾書", "其他文件", "聯絡人名片", "銀行存摺", "設立證明")
    if preparer and any(token in preparer for token in attachment_noise):
        recovered = _between(text, ("填表人",), ("填表日期",))
        recovered = _clean_cjk_spacing(recovered)
        if recovered and len(recovered) <= 30 and not any(token in recovered for token in attachment_noise):
            preparer = recovered
        else:
            preparer = ""
            warnings.append("填表人欄位疑似讀到附件清單文字，已清空，請人工確認。")
        out["填表人"] = preparer

    company = out.get("公司全名(中文)", "")

    # V2.3.5.3: when filename + preparer + bank account holder independently
    # agree on the same business name, prefer that consensus over a one-character
    # OCR disagreement (e.g. 永昌企業社 vs 永菖企業社).  This is intentionally
    # conservative and only fires when all three sources support the same name.
    bank_holder = out.get("匯款帳戶戶名", "") or ""
    consensus_name = ""
    if preparer and any(s in preparer for s in ("公司", "企業社", "行", "商號")):
        if preparer in fname and bank_holder.startswith(preparer):
            consensus_name = preparer
    if company and consensus_name and company != consensus_name:
        # Require broadly the same business-name shape so an unrelated preparer
        # cannot replace the OCR company name.
        suffixes = ("股份有限公司", "有限公司", "企業社", "商號", "工程行", "行")
        same_suffix = any(company.endswith(x) and consensus_name.endswith(x) for x in suffixes)
        if same_suffix:
            out["公司全名(中文)"] = consensus_name
            company = consensus_name

    candidates = [x for x in (fname, preparer) if x and any(s in x for s in ("公司", "企業社", "行", "商號"))]
    if company and candidates and all(company != c for c in candidates):
        warnings.append(f"公司名稱 OCR 為「{company}」，但檔名/填表人出現「{' / '.join(candidates)}」，請人工確認公司名稱。")

    return out, warnings

# ---- V2.3.4: multi-page page selection + label-anchor fallback -----------------
SUPPLIER_PAGE_MARKERS = (
    "供應商資料表", "公司全名", "統一編號", "服務範疇", "匯款帳戶", "發票類型", "課稅別"
)


def supplier_page_score(text: str) -> int:
    """Score whether OCR text looks like the GreenHarvest supplier master form."""
    t = _clean_space(text)
    score = sum(2 for marker in SUPPLIER_PAGE_MARKERS if marker in t)
    if "供應商風險評估表" in t:
        score -= 8
    if "附件1" in t and "附件5" in t:
        score += 2
    return score


def _area(text: str, starts: tuple[str, ...], ends: tuple[str, ...]) -> str:
    return _between(_clean_space(text), starts, ends)


def _first_date(value: str) -> str:
    patterns = [
        r"(?<!\d)(\d{2,4}\s*[./年]\s*\d{1,2}\s*[./月]\s*\d{1,2}\s*日?)(?!\d)",
        r"(?<!\d)(\d{2,4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)(?!\d)",
    ]
    for p in patterns:
        m = re.search(p, value)
        if m:
            return re.sub(r"\s+", "", m.group(1))
    return ""


def _looks_like_address(value: str) -> bool:
    return bool(value and re.search(r"[縣市區鄉鎮村里路街巷弄號樓]|Road|Rd\.|Street|St\.", value, re.I))


def _looks_like_person(value: str) -> bool:
    compact = re.sub(r"\s+", "", value or "")
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,6}", compact))


def _looks_like_bank(value: str) -> bool:
    return bool(value and ("銀行" in value or "郵局" in value or re.search(r"Bank", value, re.I))) and not bool(re.search(r"\d{6,}", value))


def _clean_label_prefix(value: str, labels: tuple[str, ...]) -> str:
    v = _clean_space(value)
    for label in labels:
        v = re.sub(r"^" + re.escape(label) + r"\s*[:：#]?\s*", "", v, flags=re.I)
    return v.strip(" ：:#|")


def apply_anchor_fallback(fields: dict, full_text: str) -> tuple[dict, list[str]]:
    """Use label-to-label OCR regions when fixed crops fail validation.

    This intentionally prefers blank + warning over copying a clearly wrong neighbouring
    field. It is designed for scans whose scale/row heights differ from the calibration PDF.
    """
    out = dict(fields)
    warnings: list[str] = []
    text = _clean_space(full_text)
    if not text:
        return out, warnings

    # English name.
    en = out.get("公司全名(英文)", "")
    if not en or re.search(r"中文|公司全名", en) or re.search(r"[\u4e00-\u9fff]{5,}", en):
        a = _area(text, ("英文#", "英文＃", "英文"), ("統一編號/身份證號", "統一編號", "服務範疇"))
        a = re.sub(r"(?<!\d)\d{8}(?!\d).*", "", a).strip()
        a = _clean_label_prefix(a, ("英文#", "英文＃", "英文"))
        if a and not re.search(r"[\u4e00-\u9fff]{5,}", a):
            out["公司全名(英文)"] = a

    # Service scope: a long business-code row is valid; labels/dates are not.
    scope = out.get("服務範疇", "")
    if not scope or scope.startswith("英文") or (out.get("公司全名(英文)") and out.get("公司全名(英文)") in scope):
        a = _area(text, ("服務範疇",), ("核准設立日期", "資本額"))
        a = _clean_label_prefix(a, ("服務範疇",))
        if a:
            out["服務範疇"] = _clean_cjk_spacing(a)

    # Establishment date must actually be a date.
    date = _first_date(out.get("核准設立日期(年/月/日)", ""))
    if not date:
        date = _first_date(_area(text, ("核准設立日期",), ("資本額", "聯絡地址")))
    if date:
        out["核准設立日期(年/月/日)"] = date
    else:
        out["核准設立日期(年/月/日)"] = ""
        warnings.append("核准設立日期未通過日期格式檢查，已留空，請人工確認。")

    # Capital: accept explicit 萬 / 萬元 only.
    capital = out.get("資本額(單位:萬元)", "")
    cm = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*萬(?:元)?(?:整)?", capital)
    if not cm:
        cap_area = _area(text, ("資本額",), ("聯絡地址", "帳單地址", "負責人"))
        cm = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*萬(?:元)?(?:整)?", cap_area)
    if cm:
        out["資本額(單位:萬元)"] = re.sub(r"\s+", "", cm.group(0))
    else:
        out["資本額(單位:萬元)"] = ""
        warnings.append("資本額未通過金額格式檢查，已留空，請人工確認。")

    # Addresses.
    for key, starts, ends in (
        ("聯絡地址", ("聯絡地址",), ("帳單地址", "負責人", "公司電話")),
        ("帳單地址", ("帳單地址",), ("負責人", "公司電話", "聯絡人")),
    ):
        value = out.get(key, "")
        if not _looks_like_address(value) or "單位:萬元" in value or "單位：萬元" in value:
            a = _clean_label_prefix(_area(text, starts, ends), starts)
            a = re.sub(r"[（(]?單位\s*[:：]?\s*萬元[）)]?", "", a)
            if _looks_like_address(a):
                out[key] = _clean_cjk_spacing(a)
            elif not _looks_like_address(value):
                out[key] = ""
                warnings.append(f"{key}未通過地址格式檢查，已留空，請人工確認。")

    # Responsible person / contact person.
    if not _looks_like_person(out.get("負責人", "")):
        a = _clean_label_prefix(_area(text, ("負責人",), ("公司電話", "聯絡人")), ("負責人",))
        # Stop if the OCR row also contains the company-phone label.
        a = re.split(r"公司電話", a)[0].strip()
        if _looks_like_person(a):
            out["負責人"] = _clean_person_name(a)
        else:
            out["負責人"] = ""
            warnings.append("負責人欄位疑似錯位，已留空，請人工確認。")
    if not _looks_like_person(out.get("聯絡人", "")):
        a = _clean_label_prefix(_area(text, ("聯絡人",), ("連絡電話", "聯絡電話", "匯款帳戶")), ("聯絡人",))
        a = re.split(r"連絡電話|聯絡電話", a)[0].strip()
        if _looks_like_person(a):
            out["聯絡人"] = _clean_person_name(a)

    # Banking fields. Recover only from their own label regions.
    bank = out.get("匯款銀行", "")
    if not _looks_like_bank(bank):
        a = _clean_label_prefix(_area(text, ("匯款銀行",), ("分行別", "匯款帳號")), ("匯款銀行",))
        if _looks_like_bank(a):
            out["匯款銀行"] = a
        else:
            out["匯款銀行"] = ""
            warnings.append("匯款銀行欄位疑似錯位，已留空，請人工確認。")

    branch = out.get("分行別", "")
    if not branch or re.search(r"\d{6,}|發票類型|課稅別", branch):
        a = _clean_label_prefix(_area(text, ("分行別",), ("匯款帳號", "發票類型")), ("分行別",))
        # Keep only a compact branch name when supported.
        m = re.search(r"([\u4e00-\u9fff]{1,10}(?:分行|支行|郵局))", a)
        if m:
            out["分行別"] = m.group(1)

    acct = out.get("匯款帳號", "")
    if not re.search(r"\d{6,}", acct) or "發票類型" in acct or "課稅別" in acct:
        a = _clean_label_prefix(_area(text, ("匯款帳號",), ("發票類型", "課稅別", "附件1")), ("匯款帳號",))
        nums = re.findall(r"(?<!\d)(\d[\d\- ]{5,}\d)(?!\d)", a)
        if nums:
            out["匯款帳號"] = " / ".join(re.sub(r"\s+", "", n) for n in nums[:3])
        else:
            out["匯款帳號"] = ""
            warnings.append("匯款帳號欄位疑似錯位，已留空，請人工確認。")

    account_name = out.get("匯款帳戶戶名", "")
    if not account_name or "銀行" in account_name or "分行別" in account_name:
        a = _clean_label_prefix(_area(text, ("匯款帳戶戶名",), ("匯款銀行", "分行別", "匯款帳號")), ("匯款帳戶戶名",))
        if a:
            out["匯款帳戶戶名"] = _clean_cjk_spacing(a)

    return out, warnings


# ---- V2.3.5: visual table-row / label-aware fallback --------------------------
def _line_after_label(line: str, label: str, stop_labels=()):
    compact = re.sub(r"\s+", "", line or "")
    pos = compact.find(label)
    if pos < 0:
        return ""
    value = compact[pos + len(label):]
    cut = len(value)
    for stop in stop_labels:
        q = value.find(stop)
        if q >= 0:
            cut = min(cut, q)
    return value[:cut].strip("：:#| ")


def _row_for(lines, label: str):
    key = re.sub(r"\s+", "", label)
    for row in lines or []:
        if key in row.get("compact", re.sub(r"\s+", "", row.get("text", ""))):
            return row.get("text", "")
    return ""


def apply_table_row_fallback(fields: dict, ocr_lines_data: list) -> tuple[dict, list[str]]:
    """Recover supplier fields from the visual row containing each printed label.

    Unlike the old fixed crop, this follows OCR-detected labels. It is deliberately
    conservative: row recovery replaces values only when the row supplies a value
    that passes the field's format check.
    """
    out = dict(fields)
    warnings=[]
    lines = ocr_lines_data or []
    if not lines:
        return out, warnings

    # Date + capital share a row in the GreenHarvest supplier form.
    row = _row_for(lines, "核准設立日期")
    if row:
        d = _first_date(row)
        if d:
            out["核准設立日期(年/月/日)"] = d
        m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*萬(?:元)?(?:整)?", row)
        if m:
            out["資本額(單位:萬元)"] = re.sub(r"\s+", "", m.group(0))

    # Address rows.  Strip label and any neighbour label; require address syntax.
    for key,label,nexts in (
        ("聯絡地址","聯絡地址",("帳單地址",)),
        ("帳單地址","帳單地址",("負責人","公司電話")),
    ):
        row=_row_for(lines,label)
        if row:
            v=_line_after_label(row,label,nexts)
            v=re.sub(r"[（(]?單位\s*[:：]?\s*萬元[）)]?", "", v)
            v=_clean_cjk_spacing(v)
            if _looks_like_address(v):
                out[key]=v

    # Responsible person + company phone.
    row=_row_for(lines,"負責人")
    if row:
        v=_line_after_label(row,"負責人",("公司電話",))
        v=_clean_person_name(v)
        if _looks_like_person(v): out["負責人"]=v
        m=re.search(r"(?:公司電話)\s*[:：]?\s*([0-9()\- ]{7,20})", row)
        if m: out["公司電話"]=re.sub(r"\s+", "", m.group(1)).strip("- ")

    # Contact person + phone.  Remove a leaked first character of the next label.
    row=_row_for(lines,"聯絡人")
    if row:
        v=_line_after_label(row,"聯絡人",("連絡電話","聯絡電話"))
        v=_clean_person_name(v)
        if _looks_like_person(v): out["聯絡人"]=v
        m=re.search(r"(?:連絡電話|聯絡電話)\s*[:：]?\s*([0-9()\- ]{7,20})", row)
        if m: out["連絡電話"]=re.sub(r"\s+", "", m.group(1)).strip("- ")

    # Bank account name row.
    row=_row_for(lines,"匯款帳戶戶名")
    if row:
        v=_line_after_label(row,"匯款帳戶戶名",("匯款銀行","分行別"))
        v=_clean_cjk_spacing(v)
        if v and "銀行" not in v and "分行別" not in v:
            out["匯款帳戶戶名"]=v

    # Bank + branch share a row.
    row=_row_for(lines,"匯款銀行")
    if row:
        bank=_line_after_label(row,"匯款銀行",("分行別","匯款帳號"))
        if _looks_like_bank(bank): out["匯款銀行"]=bank
        bm=re.search(r"分行別\s*[:：]?\s*([\u4e00-\u9fff]{1,10}(?:分行|支行|郵局))", row)
        if bm: out["分行別"]=bm.group(1)

    # Account row can contain multiple currency accounts.
    row=_row_for(lines,"匯款帳號")
    if row:
        # Prefer explicitly hyphenated account numbers so two adjacent accounts
        # do not get merged into one long match. Fall back to plain long digits.
        nums = re.findall(r"(?<!\d)(\d{2,4}(?:-\d{2,6}){2,4})(?!\d)", row)
        if not nums:
            nums = re.findall(r"(?<!\d)(\d{6,20})(?!\d)", row)
        nums=[re.sub(r"\s+", "", x) for x in nums]
        if nums: out["匯款帳號"]=" / ".join(dict.fromkeys(nums))

    # Service scope spans multiple visual lines between its label and the date row.
    start=None; end=None
    for i,r in enumerate(lines):
        c=r.get("compact","")
        if start is None and "服務範疇" in c: start=i
        if start is not None and i>start and "核准設立日期" in c:
            end=i; break
    if start is not None:
        chunks=[]
        for r in lines[start:(end if end is not None else min(len(lines),start+8))]:
            t=r.get("text","")
            t=re.sub(r"服務\s*範疇", "", t)
            t=re.sub(r"^(英文[#＃]?|中文)\s*[:：#]?", "", t)
            if t.strip(): chunks.append(t.strip())
        candidate=_clean_cjk_spacing(" ".join(chunks)).strip(" ,，;；|")
        # Business scope usually has codes or multiple CJK words; avoid header bleed.
        if candidate and (re.search(r"[A-Z]\d{6}", candidate) or len(candidate)>12):
            out["服務範疇"]=candidate

    # Final conservative cleanup for a common OCR artefact: a single leaked '連'
    # from the following 連絡電話 label after a 2-4 char Chinese contact name.
    cp=_clean_person_name(out.get("聯絡人", ""))
    if re.fullmatch(r"[\u4e00-\u9fff]{3,5}連", cp):
        cp=cp[:-1]
    if _looks_like_person(cp): out["聯絡人"]=cp

    # Revalidate fields that must never carry neighbouring labels.
    for key in ("聯絡地址","帳單地址"):
        v=out.get(key,"")
        v=re.sub(r"^\d{2,4}[./年]\d{1,2}[./月]\d{1,2}日?", "", v).strip()
        v=re.sub(r"[（(]?單位\s*[:：]?\s*萬元[）)]?", "", v).strip()
        if _looks_like_address(v): out[key]=v
    return out, warnings

def refine_v2352_fields(fields: dict, full_text: str = "") -> tuple[dict, list[str]]:
    """Final V2.3.5.2 convergence pass.

    Fixes four issues observed on the Fuji Bridex multi-page sample without
    disturbing fields already recovered correctly by the row-aware pass:
    1) recover establishment date from first-page OCR even when the printed
       label is split/reordered by Vision,
    2) stop service scope before the next form section,
    3) preserve all bank accounts instead of overwriting a multi-account value
       with only the first visually detected account,
    4) reconcile stale warnings after later fallbacks successfully recovered a field.
    """
    out = dict(fields)
    warnings: list[str] = []
    text = _clean_space(full_text)

    # 1) Establishment date.  Some scans OCR the label as
    # "核准 (年/月 設立/日) 日期".  The date value itself is still reliable.
    if not _first_date(out.get("核准設立日期(年/月/日)", "")):
        date_candidates = []
        for m in re.finditer(r"(?<!\d)(\d{2,4}\s*[./年]\s*\d{1,2}\s*[./月]\s*\d{1,2}\s*日?)(?!\d)", text):
            raw = m.group(1)
            # Prefer dates near the 核准/設立/日期 labels; otherwise keep as fallback.
            left = text[max(0, m.start()-80):m.start()]
            score = sum(k in left for k in ("核准", "設立", "日期"))
            date_candidates.append((score, m.start(), re.sub(r"\s+", "", raw)))
        if date_candidates:
            date_candidates.sort(key=lambda x: (-x[0], x[1]))
            out["核准設立日期(年/月/日)"] = date_candidates[0][2]

    # 2) Service scope.  Prefer the section bounded by 服務範疇 and the next
    # structural marker.  This avoids swallowing date/address/contact rows.
    scope = out.get("服務範疇", "") or ""
    scope = re.split(r"核准|資本額|聯絡地址|帳單地址|負責人|公司電話", scope, maxsplit=1)[0]
    scope = _clean_cjk_spacing(scope).strip(" ,，;；|:#")

    robust = ""
    m_start = re.search(r"服務\s*範疇", text)
    if m_start:
        tail = text[m_start.end():]
        end_positions = []
        for pat in (r"核准", r"資本\s*額", r"聯絡\s*地址", r"帳單\s*地址", r"負責人"):
            mm = re.search(pat, tail)
            if mm:
                end_positions.append(mm.start())
        if end_positions:
            tail = tail[:min(end_positions)]
        tail = re.sub(r"^(英文[#＃]?|中文)\s*[:：#|]?\s*", "", tail, flags=re.I)
        robust = _clean_cjk_spacing(tail).strip(" ,，;；|:#")
    code_pat = r"(?:[A-Z]{1,3}\d{5,7}|ZZ\d{5,7})"
    if robust:
        if len(re.findall(code_pat, robust)) >= len(re.findall(code_pat, scope)):
            scope = robust
    out["服務範疇"] = scope

    # 3) Preserve multiple bank accounts.  The anchor fallback may already have
    # two accounts while the visual row contains only the first line.
    acct_sources = [out.get("匯款帳號", "") or ""]
    if text:
        # Capture the whole account section up to invoice/tax labels.
        mm = re.search(r"匯款\s*帳號", text)
        if mm:
            tail = text[mm.end():]
            cuts = []
            for pat in (r"發票\s*類型", r"課稅\s*別", r"附件\s*1"):
                em = re.search(pat, tail)
                if em:
                    cuts.append(em.start())
            if cuts:
                tail = tail[:min(cuts)]
            acct_sources.append(tail)
    accounts = []
    for src in acct_sources:
        # Hyphenated bank-account forms first; allow 2-5 groups.
        for a in re.findall(r"(?<!\d)(\d{2,5}(?:-\d{1,8}){1,5})(?!\d)", src):
            a = re.sub(r"\s+", "", a)
            if a not in accounts:
                accounts.append(a)
        # Keep plain long digit account if no hyphenated values were found there.
        if not accounts:
            for a in re.findall(r"(?<!\d)(\d{7,20})(?!\d)", src):
                if a not in accounts:
                    accounts.append(a)
    if accounts:
        # V2.3.5.3: remove OCR fragments that are only a suffix/sub-string of a
        # complete account already found.  Keep genuinely different accounts
        # (e.g. TWD and FX accounts on the Fuji Bridex form).
        def _digits(v: str) -> str:
            return re.sub(r"\D", "", v)
        deduped = []
        for a in accounts:
            da = _digits(a)
            is_fragment = False
            for b in accounts:
                if a == b:
                    continue
                db = _digits(b)
                if len(db) > len(da) and len(da) >= 6 and da in db:
                    is_fragment = True
                    break
            if not is_fragment and a not in deduped:
                deduped.append(a)
        if deduped:
            out["匯款帳號"] = " / ".join(deduped[:3])

    return out, warnings



def finalize_company_consensus(fields: dict, filename: str = "", warnings: list[str] | None = None) -> tuple[dict, list[str]]:
    """Final company-name reconciliation after all OCR fallbacks have run.

    V2.3.5.4 moves the consensus decision to the end of the pipeline because
    bank-account holder / preparer values may themselves be recovered by later
    row-aware fallbacks. Only correct a one-character OCR disagreement when
    filename + preparer + bank holder all support the same business name.
    """
    out = dict(fields)
    ws = list(warnings or [])
    company = _clean_cjk_spacing(out.get("公司全名(中文)", "") or "")
    preparer = _clean_cjk_spacing(out.get("填表人", "") or "")
    holder = _clean_cjk_spacing(out.get("匯款帳戶戶名", "") or "")
    fname = _clean_cjk_spacing(Path(filename).stem if filename else "")

    suffixes = ("股份有限公司", "有限公司", "企業社", "商號", "工程行", "行")

    def edit_distance_le1(a: str, b: str) -> bool:
        if a == b:
            return True
        if abs(len(a) - len(b)) > 1:
            return False
        # Same length: at most one substitution.
        if len(a) == len(b):
            return sum(x != y for x, y in zip(a, b)) <= 1
        # One insertion/deletion.
        if len(a) > len(b):
            a, b = b, a
        i = j = diff = 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                i += 1; j += 1
            else:
                diff += 1; j += 1
                if diff > 1:
                    return False
        return True

    consensus = ""
    if preparer and any(preparer.endswith(s) for s in suffixes):
        if preparer in fname and holder.startswith(preparer):
            consensus = preparer

    if company and consensus and company != consensus:
        same_suffix = any(company.endswith(s) and consensus.endswith(s) for s in suffixes)
        if same_suffix and edit_distance_le1(company, consensus):
            out["公司全名(中文)"] = consensus
            # Remove stale mismatch warnings created before the final recovery.
            ws = [w for w in ws if "公司名稱 OCR 為" not in w]
            ws.append(f"公司名稱已依檔名、填表人及匯款戶名三方一致結果，由「{company}」校正為「{consensus}」。")
        elif all(company != c for c in (fname, preparer) if c):
            # Keep only one mismatch warning.
            ws = [w for w in ws if "公司名稱 OCR 為" not in w]
            ws.append(f"公司名稱 OCR 為「{company}」，但檔名/填表人出現「{fname} / {preparer}」，請人工確認公司名稱。")

    return out, list(dict.fromkeys(ws))

def reconcile_final_warnings(fields: dict, warnings: list[str]) -> list[str]:
    """Drop validation warnings that became stale after later recovery passes."""
    kept = []
    for w in warnings:
        if "核准設立日期未通過" in w and _first_date(fields.get("核准設立日期(年/月/日)", "")):
            continue
        if "聯絡地址未通過" in w and _looks_like_address(fields.get("聯絡地址", "")):
            continue
        if "帳單地址未通過" in w and _looks_like_address(fields.get("帳單地址", "")):
            continue
        if "負責人欄位疑似錯位" in w and _looks_like_person(fields.get("負責人", "")):
            continue
        if "匯款銀行欄位疑似錯位" in w and _looks_like_bank(fields.get("匯款銀行", "")):
            continue
        if "匯款帳號欄位疑似錯位" in w and re.search(r"\d{6,}", fields.get("匯款帳號", "")):
            continue
        kept.append(w)
    # Deduplicate while preserving order.
    return list(dict.fromkeys(kept))

