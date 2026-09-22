# Running the project without SQL Server

The knowledge base has **no dependency on SQL Server**. A tenant is just an
integer, so the whole admin-upload → index → ask flow works before the database
is reachable.

## Why this works

`app/db/session.py` uses `Trusted_Connection=yes` (Windows integrated auth),
which cannot authenticate from macOS, and the repo has no migrations, so the
tables do not exist anywhere. Rather than block on that, the RAG API resolves
the tenant itself:

| `RAG_AUTH_MODE` | `company_id` comes from | needs the database |
|---|---|---|
| `dev` *(current)* | the `X-Company-Id` request header | no |
| `jwt` | the authenticated user's row | yes |

Switching later is a config change, not a code change. In both modes
`company_id` is resolved **server-side** — a value in a request body is always
ignored.

## Start it

```bash
cd /Users/tokamohamed/Downloads/ai_contact_center
.venv/bin/uvicorn rag.api.app:app --reload --port 8100 --app-dir rag/src
```

| URL | what it is |
|---|---|
| <http://127.0.0.1:8100/ui/kb.html> | admin upload + agent console |
| <http://127.0.0.1:8100/docs> | interactive API docs |
| <http://127.0.0.1:8100/health> | Chroma Cloud + embedding status |

Required in `rag/.env`: `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE`,
`HF_API_TOKEN` (embeddings) and `GEMINI_API_KEY` (generation).

## Use it

In the browser: set **Company ID**, drop a PDF, then ask a question. Everything
is scoped to that company id.

Or over HTTP:

```bash
# upload
curl -X POST localhost:8100/api/v1/kb/documents \
  -H "X-Company-Id: 7" -F "file=@./manual.pdf"

# list
curl localhost:8100/api/v1/kb/documents -H "X-Company-Id: 7"

# ask
curl -X POST localhost:8100/api/v1/kb/ask \
  -H "X-Company-Id: 7" -H 'Content-Type: application/json' \
  -d '{"query":"لو رجّعت حاجة بالتقسيط، الفلوس ترجع إزاي؟","top_k":4}'

# delete
curl -X DELETE localhost:8100/api/v1/kb/documents/<document_id> \
  -H "X-Company-Id: 7"
```

Or entirely from the CLI, no server:

```bash
rag-cli kb inspect --file ./manual.pdf          # extraction quality, before indexing
rag-cli kb ingest  --company-id 7 --path ./docs --export chunks.json
rag-cli kb ask     --company-id 7 -q "your question"
rag-cli kb list    --company-id 7
rag-cli kb delete  --company-id 7 --document-id <id>
```

## Running the main app too (optional)

`app/main.py` boots without a database — the engine connects lazily, so
`/health` and the schema load fine and only the DB-backed routes fail:

```bash
.venv/bin/uvicorn app.main:app --reload --port 8000
```

To reach a real SQL Server from macOS you need three things, not just
credentials:

1. **The ODBC driver.** `pyodbc.drivers()` currently returns `[]`.
   `brew tap microsoft/mssql-release && brew trust microsoft/mssql-release &&
   brew install msodbcsql18` (the tap must be explicitly trusted).
2. **SQL authentication instead of integrated auth** — `Trusted_Connection=yes`
   must become `UID=...;PWD=...`. Best done by making it conditional so
   Windows teammates keep working:

   ```python
   auth = (
       f"UID={settings.DB_USER};PWD={settings.DB_PASSWORD};"
       if settings.DB_USER else "Trusted_Connection=yes;"
   )
   ```
3. **Network access** to that host on port 1433 (VPN/firewall).

Even then the tables must already exist on that server — there are no
migrations in this repo. See `docs/multi-tenant-kb-plan.md` §3.

## Switching to JWT later

1. Set `RAG_AUTH_MODE=jwt`.
2. In `rag/src/rag/api/deps.py`, implement `_tenant_from_jwt` against
   `app.api.dependencies.get_current_user`, returning
   `Tenant(company_id=user.company_id)` and refusing a user whose
   `company_id` is `None`.
3. Mount the router inside the main app instead of running it standalone:

   ```python
   from rag.api.app import rag_router
   app.include_router(rag_router, prefix="/api/v1")
   ```

Nothing else changes: the routes already take the tenant from a dependency.
