"""
辨識工具
------------------------------------------------
文字欄位：呼叫 Google Cloud Vision API 做文字辨識(支援手寫)。
勾選欄位：不用 OCR，改用「裁切區域內黑色像素比例」判斷是否有打勾，
          比對 OCR 判斷「是不是 V」更穩定，也不受簽名字跡潦草影響。
------------------------------------------------
"""

from io import BytesIO

import numpy as np
from PIL import Image
from google.cloud import vision


def crop_field(page_image: Image.Image, box):
    x0, y0, x1, y1 = box
    return page_image.crop((x0, y0, x1, y1))


def image_to_bytes(pil_image: Image.Image) -> bytes:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def remove_table_lines(pil_image: Image.Image, min_frac: float = 0.6) -> Image.Image:
    """
    把裁切圖裡的表格框線塗白，只留下文字/手寫。
    欄位框剛好貼著格線裁切，框線常被 OCR 讀成「1」「|」「—」，
    例如得分格的「7」會變成「17」。長度超過圖寬(或圖高) min_frac 的直線、橫線才算框線，
    手寫筆畫不會這麼長又這麼直，不會被誤刪。
    """
    import cv2
    gray = np.array(pil_image.convert("L"))
    h, w = gray.shape
    if h < 10 or w < 10:
        return pil_image
    ink = ((gray < 180) * 255).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * min_frac), 10), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * min_frac), 10)))
    lines = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, hk) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, vk),
                       np.ones((3, 3), np.uint8))
    out = gray.copy()
    out[lines > 0] = 255
    return Image.fromarray(out).convert("RGB")


def prepare_digit_image(pil_image: Image.Image, target_h: int = 160, pad_ratio: float = 1.0) -> Image.Image:
    """
    分數格的手寫數字(單獨一個「7」)放大、加白邊，給 Vision 辨識。
    Vision 對「小圖裡只有一個字」常常整個不回傳；把筆跡裁到剛好、放大到約 160px 高，
    四周再留一圈跟字一樣大的白邊，就跟一般文件裡的字差不多大，辨識率會高很多。
    沒塗乾淨的框線殘段(細長、貼著格子邊)和雜點不算筆跡，免得被讀成「1」或把字縮小。
    沒有筆跡就回傳 None。
    """
    import cv2
    gray = np.array(remove_table_lines(pil_image).convert("L"))
    H, W = gray.shape
    ink = ((gray < 160) * 255).astype(np.uint8)
    # 掃描後筆畫常斷成一段一段(例如「9」的直筆變虛線)，先稍微加粗再分區塊，斷掉的筆畫才會算同一個字
    merged = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    keep = np.zeros_like(gray, dtype=bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 60:                                   # 掃描雜點(加粗後的面積)
            continue
        thin = min(w, h) <= 9                           # 加粗後細線約 5~9px 寬
        at_edge = x <= 3 or y <= 3 or x + w >= W - 3 or y + h >= H - 3
        if thin and (at_edge or w >= W * 0.5 or h >= H * 0.6):   # 沒塗乾淨的框線殘段
            continue
        keep |= labels == i
    keep &= ink > 0
    ys, xs = np.where(keep)
    if len(xs) < 15:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    glyph = Image.fromarray(np.where(keep, gray, 255).astype(np.uint8)[y0:y1, x0:x1])
    scale = target_h / max(1, glyph.height)
    glyph = glyph.resize((max(1, int(glyph.width * scale)), target_h), Image.LANCZOS)
    # 筆畫加粗：把掃描斷開的小縫補起來，細筆跡也比較好認
    from PIL import ImageFilter
    glyph = glyph.filter(ImageFilter.MinFilter(7))
    pad = int(target_h * pad_ratio)
    canvas = Image.new("L", (glyph.width + 2 * pad, glyph.height + 2 * pad), 255)
    canvas.paste(glyph, (pad, pad))
    return canvas.convert("RGB")


def _vision_text(vision_client, pil_image: Image.Image, hints, document=True) -> str:
    image = vision.Image(content=image_to_bytes(pil_image))
    context = {"language_hints": hints}
    if document:
        response = vision_client.document_text_detection(image=image, image_context=context)
    else:
        response = vision_client.text_detection(image=image, image_context=context)
    if response.error.message:
        raise RuntimeError(f"Vision API 錯誤: {response.error.message}")
    if document:
        text = response.full_text_annotation.text if response.full_text_annotation else ""
    else:
        text = response.text_annotations[0].description if response.text_annotations else ""
    return text.strip().replace("\n", " ")


def ocr_text(vision_client, pil_image: Image.Image, numeric: bool = False) -> str:
    """
    呼叫 Vision API 的文件文字辨識(document_text_detection)，適合手寫與整段文字。
    numeric=True(分數欄位)：先把手寫數字放大加白邊、改用英文數字提示；
    還是讀不到，再改用 text_detection 試一次，最後才退回原本中文模式。
    """
    if numeric:
        digit_img = prepare_digit_image(pil_image)
        if digit_img is None:
            return ""
        first = ""
        for document in (True, False):
            text = _vision_text(vision_client, digit_img, ["en"], document=document)
            if any(c.isdigit() for c in text):
                return text
            first = first or text
        if first:
            return first    # 例如把「9」讀成「q」，交給 clean_number 換回數字
    return _vision_text(vision_client, remove_table_lines(pil_image), ["zh-TW", "zh"])


def _word_text(word) -> str:
    return "".join(symbol.text for symbol in word.symbols)


def _box_from_vertices(vertices) -> list:
    xs = [v.x for v in vertices]
    ys = [v.y for v in vertices]
    return [min(xs), min(ys), max(xs), max(ys)]


def find_label_boxes(vision_client, pil_image: Image.Image, target_texts: list) -> dict:
    """
    針對整張頁面圖片跑一次完整文字辨識，找出每個目標「印刷標籤文字」
    (例如「中文」「統一編號」「服務範疇」)實際印在頁面上的位置(像素座標)。

    這是用來取代「抓格線」定位法的替代方案——格線有時候會因為掃描品質、印刷深淺
    而斷掉、消失，導致誤判；但這些標籤是印刷體文字，OCR 辨識印刷體的穩定度遠高於
    辨識斷斷續續的線條，所以改用「直接找標籤文字在哪裡」來定位，不再依賴格線本身
    是否完整。

    做法：把每個文字區塊拆成一個個「字詞」，嘗試把連續幾個字詞串起來看是否剛好等於
    目標文字(這樣即使 OCR 把同一個標籤拆成好幾個字詞，也能正確組合、定位)。

    回傳 {目標文字: [x0,y0,x1,y1]}；找不到的目標文字不會出現在結果裡，
    呼叫端要自行處理「這個標籤沒找到」的狀況(例如退回原本的格線校正方式)。
    """
    content = image_to_bytes(pil_image)
    image = vision.Image(content=content)
    response = vision_client.document_text_detection(
        image=image,
        image_context={"language_hints": ["zh-TW", "zh"]},
    )
    if response.error.message:
        raise RuntimeError(f"Vision API 錯誤: {response.error.message}")

    found = {}
    if not response.full_text_annotation:
        return found

    remaining = set(target_texts)
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                words = paragraph.words
                word_texts = [_word_text(w) for w in words]
                n = len(words)
                for start in range(n):
                    combined = ""
                    for end in range(start, min(start + 6, n)):
                        combined += word_texts[end]
                        # 用「前綴相符」而不是完全相等，這樣即使標籤後面還黏著其他符號
                        # (例如「英文#」OCR 辨識成同一個字詞)，一樣能正確比對到。
                        matched = next((t for t in remaining if combined == t or combined.startswith(t)), None)
                        if matched:
                            verts_x, verts_y = [], []
                            for w in words[start:end + 1]:
                                bx = _box_from_vertices(w.bounding_box.vertices)
                                verts_x.extend([bx[0], bx[2]])
                                verts_y.extend([bx[1], bx[3]])
                            found[matched] = [min(verts_x), min(verts_y), max(verts_x), max(verts_y)]
                            remaining.discard(matched)
                        if not remaining:
                            return found
    return found


def is_checked(page_image: Image.Image, box, margin: int = 6, threshold: float = 0.005):
    """
    判斷勾選格有沒有打勾，回傳 (是否勾選, 筆跡比例)。

    舊做法只算「框內黑色像素比例」，但掃描歪斜或縮小時，表格框線(或斷斷續續的虛線)
    常常會落進框裡，被誤判成有勾。新做法把框內的黑色區塊一塊一塊分開看，
    只計算「像勾號的筆跡」：高度至少 12px、寬度至少 8px 的區塊；
    細長的橫線、直線碎段(不管連續或斷續)都不算。
    """
    import cv2
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    mx, my = max(margin, int(w * 0.06)), max(margin, int(h * 0.12))
    if w - 2 * mx < 5 or h - 2 * my < 5:
        return False, 0.0
    arr = np.array(page_image.convert("L").crop((x0 + mx, y0 + my, x1 - mx, y1 - my)))
    ink = ((arr < 150) * 255).astype(np.uint8)
    H, W = ink.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    mark = 0
    for i in range(1, n):
        bw, bh, area = stats[i, 2], stats[i, 3], stats[i, 4]
        if bw >= 0.9 * W or (bh >= 0.95 * H and bw <= 6):   # 橫跨整格的框線、貼邊的直線
            continue
        if bh >= 12 and bw >= 8 and area >= 25:
            mark += area
    ratio = mark / ink.size
    return mark >= 25, ratio
