"""
判讀核心(不依賴 Streamlit)
------------------------------------------------
Streamlit 介面(streamlit_app.py)和判讀 API(api.py)共用這一份程式，
修正一次、兩邊同時生效。

包含：PDF/圖片轉頁面影像、依模板座標對齊與裁切、Vision OCR、勾選判斷、評核表分數檢查。
------------------------------------------------
"""

import json
import os
from io import BytesIO

from PIL import Image

from vision_utils import crop_field, ocr_text, ocr_digits_batch, is_checked, find_label_boxes
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



NUMERIC_SUFFIXES = ("_評分", "_得分", "_加分")
_DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "〇": "0", "Ｏ": "0", "D": "0", "l": "1", "I": "1", "|": "1", "｜": "1", "丨": "1",
                            "q": "9", "g": "9", "Z": "2", "z": "2", "S": "5", "s": "5", "b": "6", "B": "8", "T": "7", "ㄱ": "7", "$": "8", "&": "8",
                            **{chr(0xFF10 + i): str(i) for i in range(10)}, "．": "."})


def clean_number(text: str) -> str:
    """
    分數欄位只會是數字：把 OCR 常見的誤認(O→0、l/I/|→1、q/g→9、S→5、Z→2、全形數字)換回來，去掉空白和雜訊。
    清完沒有任何數字就回傳原文，讓人工核對時看得到 OCR 原本讀到什麼。
    """
    fixed = "".join(c for c in text.translate(_DIGIT_FIX) if c.isdigit() or c == ".").strip(".")
    return fixed if any(c.isdigit() for c in fixed) else text


def is_numeric_field(name: str) -> bool:
    return name.endswith(NUMERIC_SUFFIXES) or name == "評核總分"


def load_template(template_path):
    """
    讀取表單座標設定檔。這裡刻意不加 @st.cache_data——
    這個檔案很小、讀取很快，不需要快取；而且快取是用「檔案路徑字串」當依據，
    不是看檔案實際內容，曾經發生過 GitHub 上的檔案明明已經更新，
    但快取沒清乾淨、程式還在用記憶體裡舊內容的狀況，拿掉快取直接根除這個風險。
    """
    with open(template_path, encoding="utf-8") as f:
        return json.load(f)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _has_long_hline(page_img, y_range, min_width_frac):
    """頁面的 y_range 範圍內，有沒有一條橫跨 min_width_frac 頁寬以上的橫線(表格框線)。"""
    import cv2
    import numpy as np
    gray = np.array(page_img.convert("L"))
    y0, y1 = max(0, y_range[0]), min(gray.shape[0], y_range[1])
    if y1 <= y0:
        return False
    ink = ((gray[y0:y1] < 180) * 255).astype(np.uint8)
    # 先上下加粗，掃描稍微歪斜(線條左右兩端差幾個像素)時一樣連得起來
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(gray.shape[1] * min_width_frac), 1))
    return bool(cv2.erode(ink, kernel).any())


def deskew_page(page_img, max_angle=2.0):
    """
    把掃描歪斜的頁面轉正。用頁面上的橫線(表格框線)量出傾斜角度後反向旋轉。
    掃描檔常有 0.3~0.5 度的歪斜，橫跨整頁的格線左右兩端就會差 5~10px，
    細的格線在比對時會斷掉、找不到，改抓到旁邊較粗的線。
    做法：只留下橫線，在 ±max_angle 度內每 0.05 度試轉一次，挑橫線最「平」的角度。
    沒有橫線、或角度太小(<0.05 度)就原圖回傳。
    """
    import cv2
    import numpy as np
    gray = np.array(page_img.convert("L"))
    h, w = gray.shape
    ink = ((gray < 180) * 255).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1)))
    small = cv2.resize(lines, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    if small.sum() == 0:
        return page_img
    # 逐一試轉角度，橫線全部轉平時，每一列的墨色加總會最集中(平方和最大)
    center = (small.shape[1] / 2, small.shape[0] / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-6, 0.05):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small, m, (small.shape[1], small.shape[0]))
        score = float((rotated.sum(axis=1) ** 2).sum())
        if score > best_score:
            best_angle, best_score = float(angle), score
    angle = best_angle
    if abs(angle) < 0.05:
        return page_img
    # 用 NEAREST 旋轉：細格線不會被內插糊成淺灰色而在之後的比對中斷掉
    return page_img.rotate(angle, resample=Image.NEAREST, fillcolor=(255, 255, 255))


def pick_template_version(template, page_images):
    """
    同一種表單有新舊版面時(例如評核表 V1.1 把 e1 移到第 1 頁)，依這份掃描檔的版面挑對的模板。
    模板裡的 "versions" 列出其他版本與判斷方式，目前支援 detect_hline：
    指定頁面的某個 y 範圍內有一條夠長的橫線，就是那個版本。都不符合就用原本的模板。
    """
    for ver in template.get("versions", []):
        rule = ver.get("detect_hline")
        if not rule:
            continue
        idx = rule.get("page", 1) - 1
        if idx >= len(page_images):
            continue
        y_range = rule["y_range"]
        if rule.get("page_top_line") is not None:
            # 先扣掉整頁上下位移，舊版評分表底線往下移時才不會被誤認成 V1.1 的 e1 加分表
            import numpy as np
            from auto_align import _binary, page_y_offset
            dy = page_y_offset(_binary(np.array(page_images[idx].convert("L"))), rule)
            y_range = [y_range[0] + dy, y_range[1] + dy]
        if _has_long_hline(page_images[idx], y_range, rule.get("min_width_frac", 0.5)):
            return load_template(os.path.join(BASE_DIR, ver["template"]))
    return template


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
    if template.get("deskew"):
        page_images = [deskew_page(p) for p in page_images]
    template = pick_template_version(template, page_images)
    suggestions = {}
    crops = {}
    boxes_used = {}
    zone_transform_cache = {}
    numeric_crops = []
    for field in template["fields"]:
        if field["type"] == "text":
            img = get_field_page_image(page_images, field)
            box = resolve_box(field, page_images, template, zone_transform_cache, vision_client, warn)
            crop = crop_field(img, box)
            crops[field["name"]] = crop
            boxes_used[field["name"]] = box
            if is_numeric_field(field["name"]):
                # 分數欄位先保留位置，等整張表的分數格收齊後一起辨識(見下方 ocr_digits_batch)
                suggestions[field["name"]] = ""
                numeric_crops.append((field["name"], crop))
                continue
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
    _ocr_numeric_fields(vision_client, numeric_crops, suggestions, warn)
    return suggestions, crops, boxes_used


def _plausible_score(name: str, value: str) -> bool:
    """每個項目的評分/得分/加分都不會超過 10，評核總分不會超過 100；超過就是讀錯(例如 10 讀成 1100)。"""
    try:
        v = float(value)
    except ValueError:
        return False
    return v <= (100 if name == "評核總分" else 10)


def _ocr_numeric_fields(vision_client, numeric_crops, suggestions, warn=None):
    """分數欄位：整張表一次批次辨識；批次沒讀到數字的格子，再單格放大辨識一次。"""
    if not numeric_crops:
        return
    try:
        batch = ocr_digits_batch(vision_client, [crop for _, crop in numeric_crops])
    except Exception as e:
        _warn(warn, f"分數欄位批次辨識失敗，改成逐格辨識：{e}")
        batch = [""] * len(numeric_crops)
    for (name, crop), text in zip(numeric_crops, batch):
        value = clean_number(text) if text else ""
        if not any(c.isdigit() for c in value) or not _plausible_score(name, value):
            try:
                single = ocr_text(vision_client, crop, numeric=True)
                single = clean_number(single) if single else ""
                if any(c.isdigit() for c in single) and (
                        _plausible_score(name, single) or not any(c.isdigit() for c in value)):
                    value = single
            except Exception as e:
                _warn(warn, f"「{name}」辨識失敗：{e}")
        suggestions[name] = value


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
