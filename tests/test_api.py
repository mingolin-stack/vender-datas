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


def test_quotation_not_ready(client):
    r = post(client, "/api/quotation/extract", 1)
    assert r.status_code == 501 and r.json()["success"] is False


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
