"""API 路由與回傳格式測試。不呼叫 Google Vision：判讀核心用假的結果代替。"""

import importlib

import fitz
import pytest
from fastapi.testclient import TestClient

import modules.common as common


def blank_pdf(pages):
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=595, height=842)
    return doc.tobytes()


FAKE_VALUES = {
    "公司全名(中文)": "測試股份有限公司", "統一編號/身份證號": "1234567",
    "供應商名稱": "測試工程行", "a1_評分": "10", "a1_得分": "8",
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(common, "get_vision_client", lambda: None)
    monkeypatch.setattr(common, "run_extraction", lambda pages, template, vc, warn: (dict(FAKE_VALUES), {}, {}))
    monkeypatch.delenv("VENDOR_API_KEY", raising=False)
    monkeypatch.delenv("ENABLED_MODULES", raising=False)
    import api
    return TestClient(importlib.reload(api).app)


def post(client, url, pages, **form):
    return client.post(url, files={"file": ("a.pdf", blank_pdf(pages), "application/pdf")}, data=form)


def test_health(client):
    assert client.get("/health").json()["modules"] == ["supplier_master", "supplier_evaluation", "quotation"]


def test_supplier_master(client):
    r = post(client, "/api/supplier-master/extract", 1)
    body = r.json()
    assert r.status_code == 200 and body["success"] and body["module"] == "supplier_master"
    assert body["normalized"]["company_name_zh"] == "測試股份有限公司"
    assert any("8 碼" in w for w in body["warnings"])


def test_supplier_evaluation(client):
    body = post(client, "/api/supplier-evaluation/extract", 3).json()
    assert body["module"] == "supplier_evaluation"
    assert body["normalized"]["items"]["a1"] == {"max_score": "10", "score": "8", "note": ""}
    assert any("分開上傳" in w for w in body["warnings"])


def test_evaluation_needs_two_pages(client):
    r = post(client, "/api/supplier-evaluation/extract", 1)
    assert r.status_code == 400 and r.json()["success"] is False and "2 頁" in r.json()["message"]


QUOTE_TEXT = ["大成五金有限公司", "統一編號：53212345", "客戶：綠色收成股份有限公司", "報價單號：Q1015-03",
              "報價日期：115/10/15", "1  不鏽鋼螺絲  M6x20  支  1,000  5  5,000",
              "2  鍍鋅角鐵  40x40x3mm  支  20  350  7,000", "未稅合計 12,000", "營業稅 600", "含稅總計 12,600"]
ITEMS = [("不鏽鋼螺絲", "M6x20", "支", 1000, 5, 5000), ("鍍鋅角鐵", "40x40x3mm", "支", 20, 350, 7000)]


def item_tuples(body):
    return [(i["name"], i["spec"], i["unit"], i["quantity"], i["unit_price"], i["amount"])
            for i in body["normalized"]["items"]]


def text_pdf(lines):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # 每格分開寫、而且整欄由右到左寫入，模擬 PDF 內部文字順序跟畫面不同的情況
    cells = [(i, j, cell) for i, line in enumerate(lines) for j, cell in enumerate(line.split("  "))]
    for i, j, cell in sorted(cells, key=lambda c: (-c[1], c[0])):
        page.insert_text((40 + j * 75, 60 + i * 20), cell, fontname="china-t", fontsize=10)
    return doc.tobytes()


def test_quotation_text_pdf(client, monkeypatch):
    import modules.quotation as q
    monkeypatch.setattr(q, "ocr_page", lambda *a: pytest.fail("電子檔 PDF 不應呼叫 Vision"))
    r = client.post("/api/quotation/extract", files={"file": ("q.pdf", text_pdf(QUOTE_TEXT), "application/pdf")})
    body = r.json()
    assert r.status_code == 200 and body["module"] == "quotation" and body["text_source"] == "pdf_text"
    n = body["normalized"]
    assert (n["supplier_name"], n["tax_id"], n["quote_no"]) == ("大成五金有限公司", "53212345", "Q1015-03")
    assert (n["quote_date"], n["subtotal"], n["tax"], n["total"]) == ("2026-10-15", 12000, 600, 12600)
    assert item_tuples(body) == ITEMS and body["warnings"] == []


def test_quotation_scanned_uses_ocr(client, monkeypatch):
    import modules.quotation as q
    monkeypatch.setattr(q, "ocr_page", lambda vc, img: "\n".join(QUOTE_TEXT))
    body = post(client, "/api/quotation/extract", 1).json()
    assert body["text_source"] == "ocr" and body["normalized"]["total"] == 12600


def test_quotation_excel(client):
    from io import BytesIO
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    for row in [["大成五金有限公司"], ["統一編號", "53212345"], ["報價日期", "2026/10/15"],
                ["項次", "品名", "規格", "單位", "數量", "單價", "金額"],
                [1, "不鏽鋼螺絲", "M6x20", "支", 1000, 5, 5000], [2, "鍍鋅角鐵", "40x40x3mm", "支", 20, 350, 7000],
                ["未稅合計", 12000], ["營業稅", 600], ["含稅總計", 12600]]:
        ws.append(row)
    buf = BytesIO()
    wb.save(buf)
    r = client.post("/api/quotation/extract", files={"file": ("q.xlsx", buf.getvalue(), "application/octet-stream")})
    body = r.json()
    assert body["text_source"] == "excel" and body["normalized"]["total"] == 12600
    assert item_tuples(body) == ITEMS


@pytest.mark.parametrize("url", ["/api/extract", "/api/extract_vendor"])
def test_legacy_dispatch(client, url):
    assert post(client, url, 1).json()["module"] == "supplier_master"
    assert post(client, url, 2, document_type="supplier_evaluation").json()["module"] == "supplier_evaluation"
    assert post(client, url, 1, document_type="xxx").status_code == 400


def test_api_key(client, monkeypatch):
    monkeypatch.setenv("VENDOR_API_KEY", "secret")
    assert post(client, "/api/supplier-master/extract", 1).status_code == 401
    r = client.post("/api/supplier-master/extract", headers={"X-API-Key": "secret"},
                    files={"file": ("a.pdf", blank_pdf(1), "application/pdf")})
    assert r.status_code == 200


def test_single_module_service(monkeypatch):
    monkeypatch.setenv("ENABLED_MODULES", "supplier_evaluation")
    import api
    c = TestClient(importlib.reload(api).app)
    assert c.get("/health").json()["modules"] == ["supplier_evaluation"]
    assert post(c, "/api/supplier-master/extract", 1).status_code == 404
    assert post(c, "/api/extract", 1).status_code == 404
