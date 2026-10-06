# Google Drive upload mirror

KB files uploaded from the UI are still stored locally and indexed for their
company. When enabled, the original file is also copied into `fixed_files` in
the Google Drive account configured below. Drive mirroring is optional; a Drive
outage does not prevent a successful KB ingestion, and the upload response/UI
reports the mirror status.

## One-time authorization

1. In Google Cloud Console, create or select a project, enable **Google Drive
   API**, and configure an OAuth consent screen.
2. Create an OAuth client of type **Desktop app**. Download its JSON file and
   save it as `rag/secrets/google_drive_client.json` (the secrets folder is
   ignored by Git).
3. Install the project dependencies (`pip install -e rag`). From the repo
   root, run `python rag/scripts/google_drive_oauth.py`. In the browser, sign
   into `tokamohamed1072004@gmail.com` and grant the requested Drive file
   permission. The script checks the account and writes a private token file
   under `rag/secrets/`.
4. In `rag/.env`, set:

   ```dotenv
   GOOGLE_DRIVE_ENABLED=true
   GOOGLE_DRIVE_ACCOUNT_EMAIL=tokamohamed1072004@gmail.com
   GOOGLE_DRIVE_FOLDER_NAME=fixed_files
   GOOGLE_DRIVE_OAUTH_CLIENT_FILE=rag/secrets/google_drive_client.json
   GOOGLE_DRIVE_OAUTH_TOKEN_FILE=rag/secrets/google_drive_token.json
   ```

5. Restart the API. The folder is created on the first successful upload if it
   does not already exist. The UI shows a Drive link or an upload error.

The OAuth consent screen must permit the Google account to authorize the app.
For an app left in Testing mode, Google may require the account to be listed as
a test user. Do not commit either JSON credential file or paste their contents
into chat.
