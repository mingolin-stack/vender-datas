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


def ocr_text(vision_client, pil_image: Image.Image) -> str:
    """呼叫 Vision API 的文件文字辨識(document_text_detection)，適合手寫與整段文字。"""
    content = image_to_bytes(pil_image)
    image = vision.Image(content=content)
    response = vision_client.document_text_detection(
        image=image,
        image_context={"language_hints": ["zh-TW", "zh"]},
    )
    if response.error.message:
        raise RuntimeError(f"Vision API 錯誤: {response.error.message}")
    text = response.full_text_annotation.text if response.full_text_annotation else ""
    return text.strip().replace("\n", " ")


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


def is_checked(page_image: Image.Image, box, margin: int = 6, threshold: float = 0.005) -> bool:
    """裁切出勾選格，量測扣掉邊界後的黑色像素比例，超過門檻視為「已勾選」。"""
    x0, y0, x1, y1 = box
    gray = page_image.convert("L").crop((x0 + margin, y0 + margin, x1 - margin, y1 - margin))
    arr = np.array(gray)
    dark_ratio = (arr < 150).sum() / arr.size
    return dark_ratio > threshold, dark_ratio
