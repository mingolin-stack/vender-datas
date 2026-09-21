"""
Google Drive 存取工具
------------------------------------------------
負責：用服務帳號憑證連上 Google Drive，在指定資料夾內尋找、下載、新增或更新檔案。
------------------------------------------------
"""

import json
from io import BytesIO

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def get_drive_service(service_account_info: dict):
    """
    service_account_info: 從 Streamlit secrets 讀進來的服務帳號 JSON 內容(dict 格式)。
    """
    creds = service_account.Credentials.from_service_account_info(
        service_account_info, scopes=SCOPES
    )
    # cache_discovery=False：避免在雲端容器這種唯讀/暫時性檔案系統上，
    # googleapiclient 嘗試寫入本機探索快取檔案時發生額外的警告或錯誤。
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# 遇到暫時性的網路/連線問題(BrokenPipeError、連線中斷等)時，自動重試的次數。
# googleapiclient 的 execute(num_retries=...) 內建會針對這類暫時性錯誤自動重試，
# 不需要我們自己額外寫重試迴圈。
NUM_RETRIES = 3


def find_file_id(service, folder_id: str, filename: str):
    """在指定資料夾裡尋找檔名完全相符的檔案，回傳 file_id，找不到回傳 None。"""
    safe_name = filename.replace("'", "\\'")
    query = f"'{folder_id}' in parents and name = '{safe_name}' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=NUM_RETRIES)
    files = results.get("files", [])
    return files[0]["id"] if files else None


def find_or_create_folder(service, parent_folder_id: str, folder_name: str) -> str:
    """在指定資料夾底下找子資料夾，找不到就建立一個，回傳該子資料夾的 id。"""
    safe_name = folder_name.replace("'", "\\'")
    query = (
        f"'{parent_folder_id}' in parents and name = '{safe_name}' "
        f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute(num_retries=NUM_RETRIES)
    files = results.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(body=metadata, fields="id").execute(num_retries=NUM_RETRIES)
    return folder["id"]


def download_file_bytes(service, file_id: str) -> bytes:
    request = service.files().get_media(fileId=file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk(num_retries=NUM_RETRIES)
    buffer.seek(0)
    return buffer.read()


def upload_or_update_xlsx(service, folder_id: str, filename: str, file_bytes: bytes) -> str:
    """
    把 Excel 檔案位元組資料上傳到指定資料夾。
    若同名檔案已存在就更新內容(同一個 file_id，保留檔案的分享連結、修訂歷史)，
    否則建立新檔案。回傳 file_id。
    """
    media = MediaIoBaseUpload(
        BytesIO(file_bytes),
        mimetype=XLSX_MIME,
        resumable=False,
    )
    existing_id = find_file_id(service, folder_id, filename)
    if existing_id:
        service.files().update(fileId=existing_id, media_body=media).execute(num_retries=NUM_RETRIES)
        return existing_id
    else:
        metadata = {"name": filename, "parents": [folder_id]}
        created = service.files().create(body=metadata, media_body=media, fields="id").execute(num_retries=NUM_RETRIES)
        return created["id"]
