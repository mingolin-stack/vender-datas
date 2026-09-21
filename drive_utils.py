import os
import io
import mimetypes
from typing import Optional, Dict, Any, List
import streamlit as st
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload, MediaIoBaseDownload

NUM_RETRIES = 3
SCOPES = ['https://www.googleapis.com/auth/drive']


def handle_http_error(error: HttpError, action_description: str) -> None:
    """解析並顯示 Google Drive API 錯誤"""
    status_code = error.resp.status
    try:
        error_details = error.content.decode("utf-8")
    except Exception:
        error_details = str(error)

    st.error(f"❌ {action_description} 失敗 [HTTP Status: {status_code}]")
    st.code(error_details, language="json")


def get_drive_service():
    """從 st.secrets 初始化並取得 Google Drive API Service 物件"""
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        else:
            creds_dict = dict(st.secrets)

        credentials = Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        st.error(f"初始化 Google Drive Service 失敗: {e}")
        raise e


def find_or_create_folder(service, parent_folder_id: str, folder_name: str) -> Optional[str]:
    """尋找指定父資料夾下的同名資料夾；若不存在則建立"""
    safe_folder_name = folder_name.replace("'", "\'")
    query = (
        f"'{parent_folder_id}' in parents and "
        f"name = '{safe_folder_name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )

    try:
        results = (
            service.files()
            .list(
                q=query,
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute(num_retries=NUM_RETRIES)
        )

        files = results.get("files", [])
        if files:
            return files[0]["id"]

        folder_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }
        folder = (
            service.files()
            .create(body=folder_metadata, fields="id", supportsAllDrives=True)
            .execute(num_retries=NUM_RETRIES)
        )

        return folder.get("id")

    except HttpError as error:
        handle_http_error(error, f"搜尋或建立資料夾 '{folder_name}'")
        raise error


def search_file(service, parent_folder_id: str, file_name: str) -> Optional[Dict[str, str]]:
    """在指定資料夾中搜尋檔案並回傳包含 id 與 name 的字典"""
    safe_file_name = file_name.replace("'", "\'")
    query = (
        f"'{parent_folder_id}' in parents and "
        f"name = '{safe_file_name}' and "
        f"trashed = false"
    )
    try:
        results = (
            service.files()
            .list(
                q=query,
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute(num_retries=NUM_RETRIES)
        )
        files = results.get("files", [])
        return files[0] if files else None
    except HttpError as error:
        handle_http_error(error, f"搜尋檔案 '{file_name}'")
        raise error


def download_file_bytes(service, file_id: str) -> bytes:
    """根據 file_id 下載雲端檔案內容"""
    try:
        request = service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return file_stream.getvalue()
    except HttpError as error:
        handle_http_error(error, f"下載檔案 (ID: {file_id})")
        raise error


def upload_file_from_bytes(
    service,
    file_bytes: bytes,
    file_name: str,
    parent_folder_id: str,
    mime_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """上傳檔案 bytes 至 Google Drive 資料夾"""
    if not mime_type:
        mime_type, _ = mimetypes.guess_type(file_name)
        if mime_type is None:
            mime_type = "application/octet-stream"

    file_metadata = {
        "name": file_name,
        "parents": [parent_folder_id]
    }
    media_stream = io.BytesIO(file_bytes)

    try:
        media = MediaIoBaseUpload(
            media_stream, mimetype=mime_type, resumable=True
        )
        uploaded_file = (
            service.files()
            .create(
                body=file_metadata,
                media_body=media,
                fields="id, name, webViewLink, webContentLink",
                supportsAllDrives=True
            )
            .execute(num_retries=NUM_RETRIES)
        )
        return uploaded_file
    except HttpError as error:
        handle_http_error(error, f"上傳檔案 '{file_name}'")
        raise error


def upload_or_update_xlsx(service, folder_id: str, file_name: str, xlsx_bytes: bytes) -> str:
    """新建或覆蓋更新 Excel 主檔"""
    existing_file = search_file(service, folder_id, file_name)
    media = MediaIoBaseUpload(
        io.BytesIO(xlsx_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        resumable=True
    )

    try:
        if existing_file:
            updated_file = (
                service.files()
                .update(
                    fileId=existing_file["id"],
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True
                )
                .execute(num_retries=NUM_RETRIES)
            )
            return updated_file.get("id")
        else:
            file_metadata = {
                "name": file_name,
                "parents": [folder_id]
            }
            new_file = (
                service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True
                )
                .execute(num_retries=NUM_RETRIES)
            )
            return new_file.get("id")
    except HttpError as error:
        handle_http_error(error, f"上傳/更新 Excel 檔案 '{file_name}'")
        raise error
