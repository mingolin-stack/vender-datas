"""
供應商表單 PDF 掃描辨識工具 (Streamlit)
------------------------------------------------
支援兩種表單：
  - 供應商資料表：主檔用「統一編號/公司名稱」覆蓋更新(彙總表每家供應商固定一列)
  - 供應商評核表：用「供應商名稱」建立評核歷史紀錄檔，每次評核新增一列(彙總表也是每次評核新增一列，保留所有評核歷史)

流程：
  1. 選擇表單類型，上傳一份已掃描的 PDF
  2. 程式自動辨識(文字欄位用 Google Vision API，勾選框用像素判斷)
  3. 畫面顯示「裁切小圖 + 辨識建議值」，同仁快速核對、修正
  4. 按下確認，資料寫入 Google Drive 對應的主檔/評核紀錄檔 + 彙總表

需要的 Streamlit secrets(在 App 設定的 Secrets 分頁貼入)：

    drive_folder_id = "你的 Google Drive 資料夾 ID"

    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
    client_email = "...@....iam.gserviceaccount.com"
    client_id = "..."
    token_uri = "https://oauth2.googleapis.com/token"

    (把下載到的 .json 金鑰檔內容，轉成上面這種 TOML 格式貼進去即可；
     private_key 裡的換行請保留 \n)
------------------------------------------------
"""

import json
from io import BytesIO

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
from google.cloud import vision

from drive_utils import get_drive_service, find_or_create_folder, find_file_id, download_file_bytes, upload_or_update_xlsx
from vision_utils import crop_field, ocr_text, is_checked
from master_utils import build_columns, record_filename, eval_record_filename, append_row_to_workbook, upsert_row_in_summary, build_row_dict

st.set_page_config(page_title="供應商資料表 PDF 辨識工具", page_icon="🧾", layout="wide")

RENDER_DPI = 200  # 必須跟 templates/*.json 校正時使用的 DPI 一致


FORM_TYPES = {
    "供應商資料表": {
        "template_path": "templates/supplier_form.json",
        "record_key_field": "統一編號/身份證號",
        "filename_fn": "record_filename",
        "summary_mode": "upsert",
        "summary_key_column": "統一編號/身份證號",
        "supplier_subfolder": "供應商主檔",
        "summary_filename": "彙總總表.xlsx",
        "pages_per_record": 1,
    },
    "供應商評核表": {
        "template_path": "templates/eval_form.json",
        "record_key_field": "供應商名稱",
        "filename_fn": "eval_record_filename",
        "summary_mode": "append",
        "summary_key_column": None,
        "supplier_subfolder": "供應商評核紀錄",
        "summary_filename": "評核彙總表.xlsx",
        "pages_per_record": 2,  # 一份供應商評核表固定是 2 頁；一份 PDF 可能包含多個單位的評核(每個單位各佔 2 頁)
    },
}


@st.cache_data
def load_template(template_path):
    with open(template_path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def get_vision_client():
    creds_info = dict(st.secrets["gcp_service_account"])
    from google.oauth2 import service_account
    creds = service_account.Credentials.from_service_account_info(creds_info)
    return vision.ImageAnnotatorClient(credentials=creds)


@st.cache_resource
def get_drive():
    creds_info = dict(st.secrets["gcp_service_account"])
    return get_drive_service(creds_info)


def pdf_to_images(pdf_bytes: bytes, dpi: int = RENDER_DPI):
    """回傳 PDF 每一頁的圖片列表(index 0 = 第1頁)。"""
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


def run_extraction(page_images, template, vision_client):
    """對整份表格跑一次辨識，回傳 {欄位名: 建議值} 的字典，供畫面顯示與人工核對。"""
    suggestions = {}
    crops = {}
    for field in template["fields"]:
        if field["type"] == "text":
            img = get_field_page_image(page_images, field)
            crop = crop_field(img, field["box"])
            crops[field["name"]] = crop
            try:
                suggestions[field["name"]] = ocr_text(vision_client, crop)
            except Exception as e:
                suggestions[field["name"]] = ""
                st.warning(f"「{field['name']}」辨識失敗：{e}")
        elif field["type"] == "checkbox_single":
            img = get_field_page_image(page_images, field)
            crop = crop_field(img, field["box"])
            crops[field["name"]] = crop
            checked, ratio = is_checked(img, field["box"])
            suggestions[field["name"]] = "是" if checked else "否"
        elif field["type"] == "checkbox_group":
            best_label, best_ratio = None, 0
            option_crops = []
            for opt in field["options"]:
                img = get_field_page_image(page_images, opt)
                crop = crop_field(img, opt["box"])
                option_crops.append((opt["label"], crop))
                checked, ratio = is_checked(img, opt["box"])
                if checked and ratio > best_ratio:
                    best_label, best_ratio = opt["label"], ratio
            crops[field["name"]] = option_crops
            suggestions[field["name"]] = best_label or ""
    return suggestions, crops


def split_into_records(page_images, pages_per_record):
    """把整份 PDF 的頁面，依每筆紀錄固定頁數切成好幾組(例如評核表每2頁一組)。"""
    n_pages = len(page_images)
    n_records = n_pages // pages_per_record
    leftover = n_pages % pages_per_record
    records = [page_images[i * pages_per_record:(i + 1) * pages_per_record] for i in range(n_records)]
    return records, leftover


def main():
    st.title("🧾 供應商表單 PDF 掃描辨識工具")
    st.caption("上傳掃描好的供應商資料表或評核表 PDF，自動辨識後請核對，確認無誤再存檔。")

    with st.expander("🔧 Secrets 診斷工具(排除問題用，確認沒問題後可以刪掉這段)"):
        try:
            info = dict(st.secrets["gcp_service_account"])
            pk = info.get("private_key", "")
            st.write("client_email:", repr(info.get("client_email", "")))
            st.write("project_id:", repr(info.get("project_id", "")))
            st.write("private_key_id:", repr(info.get("private_key_id", "")))
            st.write("client_id:", repr(info.get("client_id", "")))
            st.write("token_uri:", repr(info.get("token_uri", "")))
            st.write("private_key 開頭 20 字元:", repr(pk[:20]))
            st.write("private_key 結尾 20 字元:", repr(pk[-20:]))
            st.write("private_key 總長度:", len(pk))
            st.write("private_key 裡「真的換行符號」數量:", pk.count("\n"))
            st.write("private_key 裡「反斜線+n 兩個字元」數量:", pk.count("\\n"))
            st.write("drive_folder_id:", repr(st.secrets.get("drive_folder_id", "")))
        except Exception as e:
            st.error(f"讀取 Secrets 時發生錯誤：{e}")

    form_type_name = st.radio("① 選擇表單類型", list(FORM_TYPES.keys()), horizontal=True, key="form_type")
    form_cfg = FORM_TYPES[form_type_name]
    template = load_template(form_cfg["template_path"])
    columns = build_columns(template)
    filename_fn = {"record_filename": record_filename, "eval_record_filename": eval_record_filename}[form_cfg["filename_fn"]]
    pages_per_record = form_cfg["pages_per_record"]

    upload_hint = f"② 上傳已掃描的{form_type_name}(PDF)"
    if pages_per_record > 1:
        upload_hint += f"，每份固定 {pages_per_record} 頁；如果同一家供應商有多個單位評核，可以把好幾份直接掃在同一份 PDF 裡，程式會自動依每 {pages_per_record} 頁切成一份份分開處理"
    uploaded_pdf = st.file_uploader(upload_hint, type=["pdf"])

    if uploaded_pdf is not None:
        pdf_bytes = uploaded_pdf.getvalue()

        cache_key = f"{form_type_name}::{uploaded_pdf.name}"
        if st.session_state.get("_current_key") != cache_key:
            # 換了表單類型或新檔案，清掉之前的暫存結果
            st.session_state["_current_key"] = cache_key
            st.session_state.pop("_records", None)

        if "_records" not in st.session_state:
            with st.spinner("辨識中，請稍候(頁數較多或網路較慢時可能需要一點時間)..."):
                page_images = pdf_to_images(pdf_bytes)
                records_images, leftover = split_into_records(page_images, pages_per_record)
                if leftover:
                    st.warning(
                        f"這份 PDF 共 {len(page_images)} 頁，除不盡每份 {pages_per_record} 頁 —— "
                        f"最後 {leftover} 頁抓不到完整的一份，已略過，請確認掃描的頁數是否正確。"
                    )
                vision_client = get_vision_client()
                records = []
                for record_images in records_images:
                    suggestions, crops = run_extraction(record_images, template, vision_client)
                    records.append({"suggestions": suggestions, "crops": crops, "saved": False})
                st.session_state["_records"] = records

        records = st.session_state["_records"]
        n = len(records)
        if n == 0:
            st.error("沒有辨識出任何一份完整的表單，請確認上傳的 PDF 頁數是否正確。")
            return

        st.success(f"辨識完成，共偵測到 {n} 份{form_type_name}，請逐份核對下方內容，確認或修正後分別存檔。")

        for idx, record in enumerate(records):
            status = "✅ 已存檔" if record["saved"] else "尚未存檔"
            with st.expander(f"第 {idx + 1} 份 / 共 {n} 份 —— {status}", expanded=not record["saved"]):
                edited = {}
                for field in template["fields"]:
                    name = field["name"]
                    col_img, col_val = st.columns([1, 2])
                    widget_key = f"in_{idx}_{name}"

                    if field["type"] == "text":
                        with col_img:
                            st.image(record["crops"][name], use_container_width=True)
                        with col_val:
                            edited[name] = st.text_input(name, value=record["suggestions"][name], key=widget_key, disabled=record["saved"])

                    elif field["type"] == "checkbox_single":
                        with col_img:
                            st.image(record["crops"][name], use_container_width=True)
                        with col_val:
                            default_yes = record["suggestions"][name] == "是"
                            checked = st.checkbox(name, value=default_yes, key=widget_key, disabled=record["saved"])
                            edited[name] = "是" if checked else "否"

                    elif field["type"] == "checkbox_group":
                        option_labels = [o["label"] for o in field["options"]]
                        suggested = record["suggestions"][name]
                        with col_img:
                            thumb_cols = st.columns(len(record["crops"][name]))
                            for tc, (label, crop) in zip(thumb_cols, record["crops"][name]):
                                with tc:
                                    st.image(crop, caption=label, use_container_width=True)
                        with col_val:
                            default_idx = option_labels.index(suggested) if suggested in option_labels else 0
                            edited[name] = st.radio(name, option_labels, index=default_idx, key=widget_key, horizontal=True, disabled=record["saved"])

                    st.divider()

                if record["saved"]:
                    st.info("這份已經存檔過了。")
                elif st.button(f"✅ 確認並存檔(第 {idx + 1} 份)", type="primary", key=f"save_{idx}"):
                    with st.spinner("寫入 Google Drive 中..."):
                        drive_folder_id = st.secrets["drive_folder_id"]
                        service = get_drive()

                        supplier_folder_id = find_or_create_folder(service, drive_folder_id, form_cfg["supplier_subfolder"])

                        source_name = f"{uploaded_pdf.name}(第{idx + 1}份)" if n > 1 else uploaded_pdf.name
                        row = build_row_dict(source_name, edited)

                        # 1) 該供應商自己的主檔/評核紀錄檔(新增一列，形成歷史紀錄)
                        fname = filename_fn(edited)
                        existing_id = find_file_id(service, supplier_folder_id, fname)
                        existing_bytes = download_file_bytes(service, existing_id) if existing_id else None
                        new_bytes = append_row_to_workbook(existing_bytes, columns, row)
                        upload_or_update_xlsx(service, supplier_folder_id, fname, new_bytes)

                        # 2) 彙總表：供應商資料表用「覆蓋更新」，評核表用「每次新增一列」(因為每次評核都是獨立事件，要保留歷史)
                        summary_name = form_cfg["summary_filename"]
                        summary_id = find_file_id(service, drive_folder_id, summary_name)
                        summary_bytes = download_file_bytes(service, summary_id) if summary_id else None
                        if form_cfg["summary_mode"] == "upsert":
                            new_summary_bytes = upsert_row_in_summary(summary_bytes, columns, row, key_column=form_cfg["summary_key_column"])
                        else:
                            new_summary_bytes = append_row_to_workbook(summary_bytes, columns, row)
                        upload_or_update_xlsx(service, drive_folder_id, summary_name, new_summary_bytes)

                    st.session_state["_records"][idx]["saved"] = True
                    st.success(f"已存檔！{form_cfg['supplier_subfolder']}/{fname}，並已同步更新 {summary_name}。")
                    st.rerun()

        if all(r["saved"] for r in records):
            st.balloons()
            st.success("這份 PDF 裡的所有評核都已經存檔完成！")


if __name__ == "__main__":
    main()
