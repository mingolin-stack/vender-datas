"""
判讀核心(不依賴 Streamlit)
------------------------------------------------
Streamlit 介面(streamlit_app.py)和判讀 API(api.py)共用這一份程式，
修正一次、兩邊同時生效。

包含：PDF/圖片轉頁面影像、依模板座標對齊與裁切、Vision OCR、勾選判斷、評核表分數檢查。
------------------------------------------------
"""

import json
from io import BytesIO

from PIL import Image

from vision_utils import crop_field, ocr_text, is_checked, find_label_boxes
from auto_align import (compute_zone_transform, remap_box, compute_row_sequence, remap_box_row_sequence,
                        refine_column_divider, ocr_labels_to_line_seed, snap_cell_x, remap_x_row_local)

RENDER_DPI = 200  # 必須跟 templates/*.json 校正時使用的 DPI 一致
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _warn(warn, msg):
    """有傳入 warn(例如 st.warning 或 list.append)就回報，沒有就忽略。"""
    if warn:
        warn(msg)


EVAL_ITEMS = ["a1", "a2", "b1", "b2", "c1", "c2", "c3", "d1", "d2", "d3"]
EVAL_BONUS = ["e1", "e2"]


def _num(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def check_eval_record(values: dict):
    """
    供應商評核表的欄位定義：
      - 「評分」= 這一項的最高給分(滿分)
      - 「得分」= 實際評核給的分數
      - 「加分」= e1/e2 額外加分，單項上限 10 分
    檢查：得分不能超過該項最高給分、得分/加分要是數字、加分不超過 10、
    以及本單位「得分合計 + 加分」是否等於表上寫的評核總分。
    回傳 (警示清單, 本單位得分合計, 本單位滿分合計)
    """
    warns, total, full = [], 0.0, 0.0
    for it in EVAL_ITEMS:
        raw_s, raw_m = str(values.get(f"{it}_得分", "")).strip(), str(values.get(f"{it}_評分", "")).strip()
        s, m = _num(raw_s), _num(raw_m)
        if raw_s and s is None:
            warns.append(f"{it} 得分「{raw_s}」不是數字")
        if raw_m and m is None:
            warns.append(f"{it} 評分(最高給分)「{raw_m}」不是數字")
        if s is not None and m is not None and s > m:
            warns.append(f"{it} 得分 {s:g} 超過這項最高給分 {m:g}")
        if s is not None:
            total += s
            full += m if m is not None else 0
    for it in EVAL_BONUS:
        raw = str(values.get(f"{it}_加分", "")).strip()
        b = _num(raw)
        if raw and b is None:
            warns.append(f"{it} 加分「{raw}」不是數字")
        elif b is not None:
            if b > 10:
                warns.append(f"{it} 加分 {b:g} 超過單項上限 10 分")
            total += min(b, 10)
    written = _num(values.get("評核總分", ""))
    if written is not None and abs(written - total) > 0.01:
        warns.append(f"表上評核總分 {written:g}，但本單位得分＋加分合計為 {total:g}，請核對")
    return warns, total, full



def load_template(template_path):
    """
    讀取表單座標設定檔。這裡刻意不加 @st.cache_data——
    這個檔案很小、讀取很快，不需要快取；而且快取是用「檔案路徑字串」當依據，
    不是看檔案實際內容，曾經發生過 GitHub 上的檔案明明已經更新，
    但快取沒清乾淨、程式還在用記憶體裡舊內容的狀況，拿掉快取直接根除這個風險。
    """
    with open(template_path, encoding="utf-8") as f:
        return json.load(f)


def pdf_to_images(pdf_bytes: bytes, dpi: int = RENDER_DPI):
    """回傳 PDF 每一頁的圖片列表(index 0 = 第1頁)。"""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return images


def get_field_page_image(page_images, field_or_option):
    """依欄位設定的 page(預設第1頁，1-indexed)取出對應頁面的圖片。"""
    page_no = field_or_option.get("page", 1)
    idx = page_no - 1
    if idx >= len(page_images):
        raise ValueError(f"這份 PDF 只有 {len(page_images)} 頁，但欄位設定要讀第 {page_no} 頁，頁數對不上。")
    return page_images[idx]


def compute_ocr_y_seed(vision_client, page_img, zone, warn=None):
    """
    用 OCR 找出 zone 設定裡指定的「印刷標籤文字」實際位置，換算成對應的 y_map 種子值。
    這是用來取代「抓格線」定位法的替代方案——格線有時候會因為掃描品質、印刷深淺
    而斷掉、消失，導致誤判；改用「直接找標籤文字在哪裡」來定位前幾條最容易出錯的
    格線，找不到的標籤就不種，讓後續邏輯退回原本的格線校正方式。

    zone 需要有 "ocr_anchors" 設定，格式為 {模板y座標: 標籤文字}，例如：
      {296: "中文", 376: "英文", 452: "服務範疇"}
    """
    ocr_anchors = zone.get("ocr_anchors")
    if not ocr_anchors or not vision_client:
        return {}
    try:
        targets = sorted({c for spec in ocr_anchors.values() for c in str(spec).split("|")})
        label_boxes = find_label_boxes(vision_client, page_img, targets)
    except Exception as e:
        _warn(warn, f"OCR 標籤定位失敗，退回原本的格線校正方式：{e}")
        return {}
    # 標籤文字頂端 ≠ 格線位置：改從標籤往上找最近的橫線當作這一列的上框線(見 auto_align.ocr_labels_to_line_seed)
    label_tops = {text: box[1] for text, box in label_boxes.items()}
    return ocr_labels_to_line_seed(page_img, zone, label_tops)


def resolve_box(field_or_option, page_images, template, zone_transform_cache, vision_client=None, warn=None):
    """
    取得欄位實際要裁切的座標。
    如果這個模板有定義 zones，而且這個欄位有指定 zone，就先在這一頁實際圖片上
    重新校正一次(每個 zone 每一頁只需要算一次，用 cache 避免重複計算)，
    把模板座標動態對應到這一頁真正的格線位置，才回傳最終座標。

    zone 的型別分兩種：
      - 一般(預設)：用區塊頭尾兩個錨點抓一個縮放比例，套用到區塊內所有欄位。
      - "rows"：列高可能不規則(例如評分表)，改用逐列往下找的方式，更準確；
                如果 zone 有設定 "ocr_anchors"，會先用 OCR 標籤定位校正前幾條
                最容易出錯的格線，其餘的才用格線偵測。
    """
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
            ocr_y_seed = compute_ocr_y_seed(vision_client, page_img, zone, warn) if zone.get("ocr_anchors") else None
            zone_transform_cache[cache_key] = compute_row_sequence(page_img, zone, ocr_y_seed=ocr_y_seed)
        result_box = remap_box_row_sequence(box, zone_transform_cache[cache_key])
    else:
        if cache_key not in zone_transform_cache:
            page_img = get_field_page_image(page_images, field_or_option)
            zone_transform_cache[cache_key] = compute_zone_transform(page_img, zone)
        result_box = remap_box(box, zone_transform_cache[cache_key])

    if zone.get("row_local_x"):
        page_img = get_field_page_image(page_images, field_or_option)
        result_box = remap_x_row_local(page_img, box, result_box, zone)

    if zone.get("snap_cell_x") and field_or_option.get("type") == "text":
        page_img = get_field_page_image(page_images, field_or_option)
        result_box = snap_cell_x(page_img, result_box)

    if field_or_option.get("column_refine"):
        page_img = get_field_page_image(page_images, field_or_option)
        result_box = refine_column_divider(page_img, result_box)

    return result_box


def run_extraction(page_images, template, vision_client, warn=None):
    """對整份表格跑一次辨識，回傳 {欄位名: 建議值} 的字典，供畫面顯示與人工核對。"""
    suggestions = {}
    crops = {}
    boxes_used = {}
    zone_transform_cache = {}
    for field in template["fields"]:
        if field["type"] == "text":
            img = get_field_page_image(page_images, field)
            box = resolve_box(field, page_images, template, zone_transform_cache, vision_client, warn)
            crop = crop_field(img, box)
            crops[field["name"]] = crop
            boxes_used[field["name"]] = box
            try:
                suggestions[field["name"]] = ocr_text(vision_client, crop)
            except Exception as e:
                suggestions[field["name"]] = ""
                _warn(warn, f"「{field['name']}」辨識失敗：{e}")
        elif field["type"] == "checkbox_single":
            img = get_field_page_image(page_images, field)
            box = resolve_box(field, page_images, template, zone_transform_cache, vision_client, warn)
            crop = crop_field(img, box)
            crops[field["name"]] = crop
            boxes_used[field["name"]] = box
            checked, ratio = is_checked(img, box)
            suggestions[field["name"]] = "是" if checked else "否"
        elif field["type"] == "checkbox_group":
            best_label, best_ratio = None, 0
            hit_labels = []
            option_crops = []
            option_boxes = []
            for opt in field["options"]:
                img = get_field_page_image(page_images, opt)
                box = resolve_box(opt, page_images, template, zone_transform_cache, vision_client, warn)
                crop = crop_field(img, box)
                option_crops.append((opt["label"], crop))
                option_boxes.append((opt["label"], box))
                checked, ratio = is_checked(img, box)
                if checked:
                    hit_labels.append(opt["label"])
                if checked and ratio > best_ratio:
                    best_label, best_ratio = opt["label"], ratio
            crops[field["name"]] = option_crops
            boxes_used[field["name"]] = option_boxes
            suggestions[field["name"]] = best_label or ""
            if len(hit_labels) > 1:
                _warn(warn, f"「{field['name']}」勾了不只一個：{'、'.join(hit_labels)}（先帶入最明顯的「{best_label}」，請核對）")
    return suggestions, crops, boxes_used



def file_to_images(file_bytes: bytes, filename: str, template: dict = None):
    """
    PDF 或圖片都轉成頁面影像。圖片會縮放成模板校正時的尺寸(A4 @ 200 DPI)，
    多頁 TIFF 每一頁各算一頁。
    """
    name = (filename or "").lower()
    if name.endswith(".pdf") or file_bytes[:4] == b"%PDF":
        return pdf_to_images(file_bytes)
    if not name.endswith(IMAGE_EXTS):
        raise ValueError(f"不支援的檔案格式：{filename}（請上傳 PDF 或圖片）")
    img = Image.open(BytesIO(file_bytes))
    size = None
    if template and template.get("page_size_px"):
        size = (template["page_size_px"]["width"], template["page_size_px"]["height"])
    pages = []
    i = 0
    while True:
        try:
            img.seek(i)
        except EOFError:
            break
        p = img.convert("RGB")
        if size and p.size != size:
            if (p.width > p.height) != (size[0] > size[1]):
                p = p.rotate(90, expand=True)
            p = p.resize(size, Image.LANCZOS)
        pages.append(p)
        i += 1
    return pages
