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
from vision_utils import crop_field, ocr_text, is_checked, find_label_boxes
from master_utils import build_columns, record_filename, eval_record_filename, append_row_to_workbook, upsert_row_in_summary, build_row_dict, aggregate_score, read_rows_from_workbook, upsert_marked_row
from auto_align import compute_zone_transform, remap_box, compute_row_sequence, remap_box_row_sequence, refine_column_divider

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
        "aggregate": False,
    },
    "供應商評核表": {
        "template_path": "templates/eval_form.json",
        "record_key_field": "供應商名稱",
        "filename_fn": "eval_record_filename",
        "summary_mode": "append",
        "summary_key_column": None,
        "supplier_subfolder": "供應商評核紀錄",
        "summary_filename": "評核彙總表.xlsx",
        "pages_per_record": 2,  # 一份供應商評核表固定是 2 頁
        "aggregate": True,  # 同一家供應商可能由好幾個單位分別上傳評核，每次存檔後自動重新平均、更新彙總分數
        "aggregate_marker_column": "評核單位",
        "aggregate_marker_prefix": "彙總(",
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


def compute_ocr_y_seed(vision_client, page_img, zone):
    """
    用 OCR 找出 zone 設定裡指定的「印刷標籤文字」實際位置，換算成對應的 y_map 種子值。
    這是用來取代「抓格線」定位法的替代方案——格線有時候會因為掃描品質、印刷深淺
    而斷掉、消失，導致誤判；改用「直接找標籤文字在哪裡」來定位前幾條最容易出錯的
    格線，找不到的標籤就不種，讓後續邏輯退回原本的格線校正方式。

    zone 需要有 "ocr_anchors" 設定，格式為 {模板y座標: 標籤文字}，例如：
      {296: "中文", 376: "英文", 452: "服務範疇"}
    """
    ocr_anchors = zone.get("ocr_anchors")
    if not ocr_anchors or not vision_client:
        return {}
    try:
        label_boxes = find_label_boxes(vision_client, page_img, list(ocr_anchors.values()))
    except Exception as e:
        st.warning(f"OCR 標籤定位失敗，退回原本的格線校正方式：{e}")
        return {}
    seed = {}
    for template_y, label_text in ocr_anchors.items():
        if label_text in label_boxes:
            seed[int(template_y)] = label_boxes[label_text][1]  # 標籤框的 y0(頂端)
    return seed


def resolve_box(field_or_option, page_images, template, zone_transform_cache, vision_client=None):
    """
    取得欄位實際要裁切的座標。
    如果這個模板有定義 zones，而且這個欄位有指定 zone，就先在這一頁實際圖片上
    重新校正一次(每個 zone 每一頁只需要算一次，用 cache 避免重複計算)，
    把模板座標動態對應到這一頁真正的格線位置，才回傳最終座標。

    zone 的型別分兩種：
      - 一般(預設)：用區塊頭尾兩個錨點抓一個縮放比例，套用到區塊內所有欄位。
      - "rows"：列高可能不規則(例如評分表)，改用逐列往下找的方式，更準確；
                如果 zone 有設定 "ocr_anchors"，會先用 OCR 標籤定位校正前幾條
                最容易出錯的格線，其餘的才用格線偵測。
    """
    box = field_or_option["box"]
    zone_name = field_or_option.get("zone")
    zones = template.get("zones")
    if not zone_name or not zones or zone_name not in zones:
        return box

    zone = zones[zone_name]
    page_no = field_or_option.get("page", 1)
    cache_key = (page_no, zone_name)

    if zone.get("type") == "rows":
        if cache_key not in zone_transform_cache:
            page_img = get_field_page_image(page_images, field_or_option)
            ocr_y_seed = compute_ocr_y_seed(vision_client, page_img, zone) if zone.get("ocr_anchors") else None
            zone_transform_cache[cache_key] = compute_row_sequence(page_img, zone, ocr_y_seed=ocr_y_seed)
        result_box = remap_box_row_sequence(box, zone_transform_cache[cache_key])
    else:
        if cache_key not in zone_transform_cache:
            page_img = get_field_page_image(page_images, field_or_option)
            zone_transform_cache[cache_key] = compute_zone_transform(page_img, zone)
        result_box = remap_box(box, zone_transform_cache[cache_key])

    if field_or_option.get("column_refine"):
        page_img = get_field_page_image(page_images, field_or_option)
        result_box = refine_column_divider(page_img, result_box)

    return result_box


def run_extraction(page_images, template, vision_client):
    """對整份表格跑一次辨識，回傳 {欄位名: 建議值} 的字典，供畫面顯示與人工核對。"""
    suggestions = {}
    crops = {}
    boxes_used = {}
    zone_transform_cache = {}
    for field in template["fields"]:
        if field["type"] == "text":
            img = get_field_page_image(page_images, field)
            box = resolve_box(field, page_images, template, zone_transform_cache, vision_client)
            crop = crop_field(img, box)
            crops[field["name"]] = crop
            boxes_used[field["name"]] = box
            try:
                suggestions[field["name"]] = ocr_text(vision_client, crop)
            except Exception as e:
                suggestions[field["name"]] = ""
                st.warning(f"「{field['name']}」辨識失敗：{e}")
        elif field["type"] == "checkbox_single":
            img = get_field_page_image(page_images, field)
            box = resolve_box(field, page_images, template, zone_transform_cache, vision_client)
            crop = crop_field(img, box)
            crops[field["name"]] = crop
            boxes_used[field["name"]] = box
            checked, ratio = is_checked(img, box)
            suggestions[field["name"]] = "是" if checked else "否"
        elif field["type"] == "checkbox_group":
            best_label, best_ratio = None, 0
            option_crops = []
            option_boxes = []
            for opt in field["options"]:
                img = get_field_page_image(page_images, opt)
                box = resolve_box(opt, page_images, template, zone_transform_cache, vision_client)
                crop = crop_field(img, box)
                option_crops.append((opt["label"], crop))
                option_boxes.append((opt["label"], box))
                checked, ratio = is_checked(img, box)
                if checked and ratio > best_ratio:
                    best_label, best_ratio = opt["label"], ratio
            crops[field["name"]] = option_crops
            boxes_used[field["name"]] = option_boxes
            suggestions[field["name"]] = best_label or ""
    return suggestions, crops, boxes_used


def main():
    st.title("🧾 供應商表單 PDF 掃描辨識工具")
    st.caption("上傳掃描好的供應商資料表或評核表 PDF，自動辨識後請核對，確認無誤再存檔。")
    st.caption("🔖 程式版本：2026-10-02-v7（公司全名/統一編號/服務範疇改用OCR標籤定位，取代容易受格線品質影響的格線偵測法）")

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

    has_zones = "zones" in template
    zone_names = list(template.get("zones", {}).keys())
    st.caption(f"🔖 目前載入的模板：`{form_cfg['template_path']}` ｜ 是否有動態校正(zones)：{'✅ 有 — ' + '、'.join(zone_names) if has_zones else '❌ 沒有(會用寫死的固定座標)'}")
    filename_fn = {"record_filename": record_filename, "eval_record_filename": eval_record_filename}[form_cfg["filename_fn"]]
    pages_per_record = form_cfg["pages_per_record"]

    upload_hint = f"② 上傳已掃描的{form_type_name}(PDF，一次一份)"
    if pages_per_record > 1:
        upload_hint += (
            f"，固定 {pages_per_record} 頁。如果同一家供應商由好幾個單位分別評核，"
            f"請每個單位各自上傳自己那 {pages_per_record} 頁，一次上傳一份即可"
        )
    uploaded_pdf = st.file_uploader(upload_hint, type=["pdf"])

    if uploaded_pdf is None:
        return

    pdf_bytes = uploaded_pdf.getvalue()
    cache_key = f"{form_type_name}::{uploaded_pdf.name}"
    if st.session_state.get("_current_key") != cache_key:
        # 換了表單類型或新檔案，清掉之前的暫存結果
        st.session_state["_current_key"] = cache_key
        st.session_state.pop("_record", None)
        st.session_state.pop("_agg_saved", None)

    if "_record" not in st.session_state:
        with st.spinner("辨識中，請稍候(網路較慢時可能需要一點時間)..."):
            page_images = pdf_to_images(pdf_bytes)
            if len(page_images) != pages_per_record:
                st.warning(
                    f"這份 PDF 共 {len(page_images)} 頁，跟{form_type_name}預期的 {pages_per_record} 頁不符，"
                    f"仍會嘗試用前 {pages_per_record} 頁辨識，請確認掃描的頁數是否正確。"
                )
            record_images = page_images[:pages_per_record]

            vision_client = get_vision_client()

            with st.expander("🔧 座標診斷工具(排除問題用，確認沒問題後可以刪掉這段)"):
                st.write("這一頁的實際圖片尺寸：", record_images[0].size)
                if "zones" in template:
                    for zone_name, zone in template["zones"].items():
                        st.write(f"— zone「{zone_name}」—")
                        if zone.get("type") == "rows":
                            page_img = record_images[zone.get("page", 1) - 1]
                            ocr_seed = compute_ocr_y_seed(vision_client, page_img, zone) if zone.get("ocr_anchors") else None
                            if zone.get("ocr_anchors"):
                                st.write("OCR 標籤定位結果(模板座標 → OCR 找到的座標)：", ocr_seed)
                            rowseq = compute_row_sequence(page_img, zone, ocr_y_seed=ocr_seed)
                            st.write("y_map(模板座標 → 實際找到的座標)：", rowseq["y_map"])
                            st.write("x0：", rowseq["x0"], "｜x_scale：", rowseq["x_scale"])
                        else:
                            zt = compute_zone_transform(record_images[zone.get("page", 1) - 1], zone)
                            st.write(zt)
                zone_cache_debug = {}
                first_text_field = next(f for f in template["fields"] if f["type"] == "text")
                first_box = resolve_box(first_text_field, record_images, template, zone_cache_debug, vision_client)
                st.write(f"「{first_text_field['name']}」實際裁切座標：", first_box, "（模板原始座標：", first_text_field["box"], "）")

            suggestions, crops, boxes_used = run_extraction(record_images, template, vision_client)
            st.session_state["_record"] = {"suggestions": suggestions, "crops": crops, "boxes_used": boxes_used, "saved": False}

    record = st.session_state["_record"]
    st.success("辨識完成，請核對下方內容，確認或修正後存檔。")

    edited = {}
    for field in template["fields"]:
        name = field["name"]
        col_img, col_val = st.columns([1, 2])
        widget_key = f"in_{name}"

        if field["type"] == "text":
            with col_img:
                st.image(record["crops"][name], use_container_width=True)
                st.caption(f"裁切座標：{record.get('boxes_used', {}).get(name)}")
            with col_val:
                edited[name] = st.text_input(name, value=record["suggestions"][name], key=widget_key, disabled=record["saved"])

        elif field["type"] == "checkbox_single":
            with col_img:
                st.image(record["crops"][name], use_container_width=True)
                st.caption(f"裁切座標：{record.get('boxes_used', {}).get(name)}")
            with col_val:
                default_yes = record["suggestions"][name] == "是"
                checked = st.checkbox(name, value=default_yes, key=widget_key, disabled=record["saved"])
                edited[name] = "是" if checked else "否"

        elif field["type"] == "checkbox_group":
            option_labels = [o["label"] for o in field["options"]]
            suggested = record["suggestions"][name]
            option_boxes = dict(record.get("boxes_used", {}).get(name, []))
            with col_img:
                thumb_cols = st.columns(len(record["crops"][name]))
                for tc, (label, crop) in zip(thumb_cols, record["crops"][name]):
                    with tc:
                        st.image(crop, caption=label, use_container_width=True)
                        st.caption(f"{option_boxes.get(label)}")
            with col_val:
                default_idx = option_labels.index(suggested) if suggested in option_labels else 0
                edited[name] = st.radio(name, option_labels, index=default_idx, key=widget_key, horizontal=True, disabled=record["saved"])

        st.divider()

    fname = filename_fn(edited if not record["saved"] else record.get("saved_data", edited))

    if record["saved"]:
        st.info("這份已經存檔過了。")
    elif st.button("✅ 確認並存檔", type="primary"):
        with st.spinner("寫入 Google Drive 中..."):
            drive_folder_id = st.secrets["drive_folder_id"]
            service = get_drive()

            supplier_folder_id = find_or_create_folder(service, drive_folder_id, form_cfg["supplier_subfolder"])

            row = build_row_dict(uploaded_pdf.name, edited)

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

        st.session_state["_record"]["saved"] = True
        st.session_state["_record"]["saved_data"] = edited
        st.success(f"已存檔！{form_cfg['supplier_subfolder']}/{fname}，並已同步更新 {summary_name}。")
        st.balloons()
        st.rerun()

    if record["saved"] and form_cfg.get("aggregate"):
        st.divider()
        st.subheader("📊 多單位評分彙整平均")
        st.caption(
            "這家供應商如果已經有其他單位也上傳過評核，這裡會自動抓出目前所有單位的評核紀錄，"
            "把分數取平均、算出總分。⚠️ 目前的平均算法是：每個評核項目(a1~d3)取「有填寫的單位」的"
            "得分平均，10 個項目的平均分數加總為總分(滿分100)，加分項目(e1、e2)同樣取平均、每項上限10分"
            "再加進總分，等級依表單上寫的門檻(90分以上A級、70~89分B級、其餘C級)判定。"
            "**這個計算規則是我依表單上的文字說明推測的，請務必確認是否符合貴公司實際的計算方式，"
            "如果不對請告訴我怎麼調整。**"
        )

        saved_data = record["saved_data"]
        marker_col = form_cfg["aggregate_marker_column"]
        marker_prefix = form_cfg["aggregate_marker_prefix"]

        with st.spinner("讀取這家供應商目前已有的評核紀錄..."):
            drive_folder_id = st.secrets["drive_folder_id"]
            service = get_drive()
            supplier_folder_id = find_or_create_folder(service, drive_folder_id, form_cfg["supplier_subfolder"])
            fname = filename_fn(saved_data)
            existing_id = find_file_id(service, supplier_folder_id, fname)
            existing_bytes = download_file_bytes(service, existing_id) if existing_id else None
            all_rows = read_rows_from_workbook(existing_bytes, columns)
            unit_rows = [
                r for r in all_rows
                if not (isinstance(r.get(marker_col), str) and r.get(marker_col).startswith(marker_prefix))
            ]

        if len(unit_rows) < 2:
            st.info(f"目前這家供應商只有 {len(unit_rows)} 個單位的評核紀錄，還沒有其他單位一起評核，暫不計算彙總(至少要有 2 個單位才需要平均)。")
        else:
            agg = aggregate_score([fname] * len(unit_rows), unit_rows)

            col1, col2, col3 = st.columns(3)
            col1.metric("評核總分", agg["評核總分"])
            col2.metric("評核等級", agg["評核等級"])
            col3.metric("參與評核單位", agg["評核單位"])

            with st.expander("查看各項目平均分數"):
                for item in ["a1", "a2", "b1", "b2", "c1", "c2", "c3", "d1", "d2", "d3"]:
                    st.write(f"{item}_得分：{agg.get(f'{item}_得分', '')}")
                for item in ["e1", "e2"]:
                    st.write(f"{item}_加分：{agg.get(f'{item}_加分', '')}")

            agg_key = f"_agg_saved::{cache_key}"
            if st.session_state.get(agg_key):
                st.info("彙總分數已經存檔過了。")
            elif st.button("✅ 確認並更新彙總分數", type="primary"):
                with st.spinner("寫入 Google Drive 中..."):
                    row = build_row_dict(f"{fname}(彙總，共{len(unit_rows)}個單位)", agg)
                    new_bytes = upsert_marked_row(existing_bytes, columns, row, marker_col, marker_prefix)
                    upload_or_update_xlsx(service, supplier_folder_id, fname, new_bytes)

                    summary_name = form_cfg["summary_filename"]
                    summary_id = find_file_id(service, drive_folder_id, summary_name)
                    summary_bytes = download_file_bytes(service, summary_id) if summary_id else None
                    new_summary_bytes = upsert_marked_row(
                        summary_bytes, columns, row, marker_col, marker_prefix, match_column="供應商名稱"
                    )
                    upload_or_update_xlsx(service, drive_folder_id, summary_name, new_summary_bytes)

                st.session_state[agg_key] = True
                st.success("彙總分數已更新！")
                st.rerun()


if __name__ == "__main__":
    main()
