import pandas as pd
import io

def process_master_excel(old_excel_bytes: bytes, new_data: dict) -> bytes:
    """
    處理供應商主檔 Excel：
    將 AI 辨識出的字典資料追加或更新至主檔中，並回傳更新後的 Excel bytes。
    """
    # 預設欄位順序
    columns = [
        "公司全名", "統一編號", "負責人", "聯絡人",
        "公司電話", "聯絡電話", "聯絡地址", "帳單地址",
        "匯款帳號戶名", "匯款銀行", "分行別", "匯款帳號"
    ]

    # 如果舊有主檔存在，則讀取舊資料
    if old_excel_bytes:
        try:
            df = pd.read_excel(io.BytesIO(old_excel_bytes))
        except Exception:
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    # 確保 DataFrame 包含所有必要欄位
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    # 將新資料轉為 DataFrame Row
    new_row = {col: new_data.get(col, "") for col in columns}
    new_df = pd.DataFrame([new_row])

    # 如果有統一編號，進行比對更新；若不存在則追加
    tax_id = new_data.get("統一編號", "").strip() if new_data.get("統一編號") else ""
    
    if tax_id and "統一編號" in df.columns and (df["統一編號"].astype(str).str.strip() == tax_id).any():
        # 更新已有紀錄
        idx = df[df["統一編號"].astype(str).str.strip() == tax_id].index[0]
        for col in columns:
            if new_data.get(col):
                df.at[idx, col] = new_data[col]
    else:
        # 追加新紀錄
        df = pd.concat([df, new_df], ignore_index=True)

    # 重新排序欄位
    df = df[columns]

    # 轉回 Excel bytes
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="供應商主檔")
    
    return output.getvalue()
