"""
主檔存取邏輯
------------------------------------------------
- 每個供應商自己的主檔：每次上傳/核對確認後「新增一列」，天然形成該供應商的歷史紀錄
  (方便日後追蹤:這家供應商什麼時候更新過資料、改了什麼)。
- 彙總總表：每個供應商永遠只有「一列」，用統一編號/身份證號當作 key，
  同一家供應商重新上傳後會覆蓋更新那一列，而不是一直往下新增重複列，方便搜尋/篩選。
------------------------------------------------
"""

from io import BytesIO
from datetime import datetime

from openpyxl import Workbook, load_workbook


def build_columns(template: dict):
    columns = []
    for f in template["fields"]:
        columns.append(f["name"])
    return columns


def record_filename(record: dict) -> str:
    supplier_id = record.get("統一編號/身份證號", "").strip()
    company = record.get("公司全名(中文)", "").strip()
    key = supplier_id or company or "未命名供應商"
    safe = "".join(c for c in key if c not in '\\/:*?"<>|')
    if supplier_id and company:
        return f"{supplier_id}_{company}.xlsx"
    return f"{safe}.xlsx"


def eval_record_filename(record: dict) -> str:
    """供應商評核表：用供應商名稱建檔(每家供應商一份評核歷史紀錄檔)。"""
    company = record.get("供應商名稱", "").strip()
    safe = "".join(c for c in company if c not in '\\/:*?"<>|') or "未命名供應商"
    return f"{safe}_評核紀錄.xlsx"


def append_row_to_workbook(existing_bytes, columns, row_dict) -> bytes:
    """把一列資料附加到既有的 Excel 內容後面(沒有就新建)，回傳新的檔案位元組資料。"""
    full_columns = ["_來源檔案", "_處理時間"] + columns
    if existing_bytes:
        wb = load_workbook(BytesIO(existing_bytes))
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "資料"
        ws.append(full_columns)

    ws.append([row_dict.get(col, "") for col in full_columns])

    out = BytesIO()
    wb.save(out)
    return out.getvalue()


def upsert_row_in_summary(existing_bytes, columns, row_dict, key_column="統一編號/身份證號") -> bytes:
    """
    彙總總表：用 key_column 判斷這家供應商是不是已經有資料了。
    有 -> 覆蓋整列；沒有 -> 新增一列。
    """
    full_columns = ["_來源檔案", "_處理時間"] + columns
    key_value = row_dict.get(key_column, "")

    if existing_bytes:
        wb = load_workbook(BytesIO(existing_bytes))
        ws = wb.active
        headers = [c.value for c in ws[1]]
        if headers != full_columns:
            # 欄位結構跟現在的 schema 不一致(例如模板改版新增了欄位)
            # 仍然依欄位名稱對應寫入，避免資料寫到錯的欄位去
            pass
        key_col_idx = headers.index(key_column) + 1 if key_column in headers else None
        target_row = None
        if key_col_idx and key_value:
            for r in range(2, ws.max_row + 1):
                if ws.cell(row=r, column=key_col_idx).value == key_value:
                    target_row = r
                    break
        if target_row:
            for idx, col in enumerate(headers, start=1):
                ws.cell(row=target_row, column=idx, value=row_dict.get(col, ""))
        else:
            ws.append([row_dict.get(col, "") for col in headers])
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "供應商彙總總表"
        ws.append(full_columns)
        ws.append([row_dict.get(col, "") for col in full_columns])

    out = BytesIO()
    wb.save(out)
    return out.getvalue()


def build_row_dict(source_filename: str, extracted: dict) -> dict:
    row = {
        "_來源檔案": source_filename,
        "_處理時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    row.update(extracted)
    return row
