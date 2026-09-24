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


def aggregate_score(source_filenames: list, records: list) -> dict:
    """
    把同一次評核裡，好幾個不同單位各自填的評分，彙整平均成一份總分：
      - 每一個評核項目(a1~d3)：把所有有填數字的單位分數取平均(沒填的單位不計入)
      - 評核總分：10 個項目的平均分數加總(滿分100)
      - 加分項目(e1,e2)：同樣取平均，每項上限10分
      - 評核等級：依照表單本身寫的門檻(90分以上A級、70~89分B級、其餘C級)判定

    records: 每個單位確認後的資料 dict 組成的 list(跟 build_row_dict 的欄位格式一樣)。
    """
    items = ["a1", "a2", "b1", "b2", "c1", "c2", "c3", "d1", "d2", "d3"]
    bonus_items = ["e1", "e2"]

    def collect_numbers(field_name):
        values = []
        for r in records:
            raw = str(r.get(field_name, "")).strip()
            try:
                values.append(float(raw))
            except ValueError:
                continue
        return values

    per_item_avg = {}
    total = 0.0
    for item in items:
        nums = collect_numbers(f"{item}_得分")
        avg = sum(nums) / len(nums) if nums else None
        per_item_avg[item] = avg
        if avg is not None:
            total += avg

    bonus_avg = {}
    for item in bonus_items:
        nums = collect_numbers(f"{item}_加分")
        avg = sum(nums) / len(nums) if nums else None
        avg = min(avg, 10) if avg is not None else None  # 單項上限10分
        bonus_avg[item] = avg
        if avg is not None:
            total += avg

    if total >= 90:
        grade = "A"
    elif total >= 70:
        grade = "B"
    else:
        grade = "C"

    result = {
        "供應商名稱": records[0].get("供應商名稱", "") if records else "",
        "評核類別": records[0].get("評核類別", "") if records else "",
        "評核單位": "彙總(" + "、".join(sorted({r.get("評核單位", "") for r in records if r.get("評核單位")})) + ")",
        "評核人員": "、".join(r.get("評核人員", "") for r in records if r.get("評核人員")),
        "評核日期": max((r.get("評核日期", "") for r in records), default=""),
        "評核總分": round(total, 1),
        "評核等級": grade,
    }
    for item in items:
        avg = per_item_avg[item]
        result[f"{item}_得分"] = "" if avg is None else round(avg, 1)
    for item in bonus_items:
        avg = bonus_avg[item]
        result[f"{item}_加分"] = "" if avg is None else round(avg, 1)

    return result


def build_row_dict(source_filename: str, extracted: dict) -> dict:
    row = {
        "_來源檔案": source_filename,
        "_處理時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    row.update(extracted)
    return row
