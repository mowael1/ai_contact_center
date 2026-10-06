"""Authorize this app to upload knowledge-base files to Google Drive.

Setup: enable the Google Drive API in a Google Cloud project, create an OAuth
Desktop app client, save its JSON to the configured client-file path, then run
``python rag/scripts/google_drive_oauth.py`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from rag.config import settings

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main() -> None:
    client_file = settings.GOOGLE_DRIVE_OAUTH_CLIENT_FILE
    token_file = settings.GOOGLE_DRIVE_OAUTH_TOKEN_FILE
    if not client_file.is_file():
        raise SystemExit(
            f"OAuth client JSON not found at {client_file}. "
            "Create a Google OAuth Desktop app client and save its JSON there."
        )

    credentials = None
    if token_file.is_file():
        credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_file), SCOPES)
        credentials = flow.run_local_server(port=0, access_type="offline", prompt="consent")

    drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
    email = drive.about().get(fields="user(emailAddress)").execute()["user"]["emailAddress"]
    expected = settings.GOOGLE_DRIVE_ACCOUNT_EMAIL.casefold()
    if email.casefold() != expected:
        raise SystemExit(
            f"Signed in as {email}; expected {settings.GOOGLE_DRIVE_ACCOUNT_EMAIL}. "
            "No token was saved. Run again and choose the intended Google account."
        )

    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding="utf-8")
    try:
        token_file.chmod(0o600)
    except OSError:
        pass
    print(f"Drive access authorized for {email}. Token saved to {token_file}.")


if __name__ == "__main__":
    main()
