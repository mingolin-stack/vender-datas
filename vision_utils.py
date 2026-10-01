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


def is_checked(page_image: Image.Image, box, margin: int = 6, threshold: float = 0.005) -> bool:
    """裁切出勾選格，量測扣掉邊界後的黑色像素比例，超過門檻視為「已勾選」。"""
    x0, y0, x1, y1 = box
    gray = page_image.convert("L").crop((x0 + margin, y0 + margin, x1 - margin, y1 - margin))
    arr = np.array(gray)
    dark_ratio = (arr < 150).sum() / arr.size
    return dark_ratio > threshold, dark_ratio
