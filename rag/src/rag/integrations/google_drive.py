"""Upload KB source files to a configured Google Drive folder using OAuth."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import settings


class GoogleDriveNotConfigured(RuntimeError):
    pass


def upload_kb_file(path: Path) -> dict[str, str] | None:
    """Mirror a source file to Drive; return None when integration is disabled.

    OAuth credentials are created out of band by ``scripts/google_drive_oauth.py``.
    The authorized account is checked before any files are written.
    """
    if not settings.GOOGLE_DRIVE_ENABLED:
        return None

    token_file = settings.GOOGLE_DRIVE_OAUTH_TOKEN_FILE
    if not token_file.is_file():
        raise GoogleDriveNotConfigured(
            "Google Drive is enabled but the OAuth token file is missing"
        )

    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise GoogleDriveNotConfigured(
            "Install the Google Drive integration dependencies"
        ) from exc

    creds = Credentials.from_authorized_user_file(str(token_file), [
        "https://www.googleapis.com/auth/drive.file",
    ])
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_file.write_text(creds.to_json(), encoding="utf-8")
    if not creds.valid:
        raise GoogleDriveNotConfigured("Google Drive OAuth token is invalid or expired")

    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    account = drive.about().get(fields="user(emailAddress)").execute()["user"]["emailAddress"]
    expected = settings.GOOGLE_DRIVE_ACCOUNT_EMAIL.casefold()
    if account.casefold() != expected:
        raise GoogleDriveNotConfigured(
            f"OAuth is authorized as {account}; expected {settings.GOOGLE_DRIVE_ACCOUNT_EMAIL}"
        )

    folder_name = settings.GOOGLE_DRIVE_FOLDER_NAME.replace("'", "\\'")
    folders: list[dict[str, Any]] = drive.files().list(
        q=(f"name = '{folder_name}' and mimeType = "
           "'application/vnd.google-apps.folder' and trashed = false"),
        spaces="drive",
        fields="files(id,name)",
        pageSize=100,
    ).execute().get("files", [])
    if folders:
        folder_id = folders[0]["id"]
    else:
        folder = drive.files().create(
            body={
                "name": settings.GOOGLE_DRIVE_FOLDER_NAME,
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
        ).execute()
        folder_id = folder["id"]

    created = drive.files().create(
        body={"name": path.name, "parents": [folder_id]},
        media_body=MediaFileUpload(str(path), resumable=True),
        fields="id,name,webViewLink",
    ).execute()
    return {"file_id": created["id"], "web_view_link": created.get("webViewLink", "")}
