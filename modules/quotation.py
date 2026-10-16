"""
模組三：日常報價比價 ← 供應商報價單

報價單每家供應商格式都不同，不能用固定座標模板，之後會改用「整頁 OCR + 表格/品項解析」。
等拿到實際報價單樣本後再實作；目前端點先佔位，回傳 501，讓 App 端可以先把串接做好。

預計回傳的 normalized 格式(App 端可先照這個設計畫面)：
{
  "supplier_name": "", "tax_id": "", "quote_date": "", "quote_no": "", "valid_until": "",
  "currency": "TWD", "tax_included": true,
  "items": [{"name": "", "spec": "", "unit": "", "quantity": 0, "unit_price": 0, "amount": 0, "note": ""}],
  "subtotal": 0, "tax": 0, "total": 0,
  "payment_terms": "", "delivery_terms": ""
}
"""

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from .common import check_key

MODULE = "quotation"
DOCUMENT_TYPE = "quotation"

router = APIRouter(tags=["日常報價比價"])


def extract(data: bytes, filename: str, template_version: str = "V1"):
    raise HTTPException(status_code=501, detail="報價單判讀尚未上線(等待報價單樣本)")


@router.post("/api/quotation/extract", summary="報價單判讀(尚未上線)")
async def extract_endpoint(file: UploadFile = File(..., description="報價單 PDF 或圖片"),
                           x_api_key: str = Header(None)):
    check_key(x_api_key)
    return extract(b"", file.filename)
