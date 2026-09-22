# Running on Supabase

The models were SQL Server-specific. They are now dialect-portable, so the same
code runs on PostgreSQL (Supabase) or SQL Server.

## What changed

| was | now |
|---|---|
| `from sqlalchemy.dialects.mssql import DATETIME2` | `app/db/types.py::UtcDateTime` — `DATETIME2` on SQL Server, `TIMESTAMP` on PostgreSQL |
| `server_default=text("SYSUTCDATETIME()")` | `server_default=utcnow()` — compiles per dialect |
| `server_default=text("1")` for booleans | `expression.true()` — `BIT 1` / `BOOLEAN true` |
| `Trusted_Connection=yes` hardcoded | used only when `DB_USER` is empty |
| no migrations at all | Alembic, with an initial revision |

`Unicode`/`UnicodeText` needed no change: SQLAlchemy already maps them to
`NVARCHAR` on SQL Server and `VARCHAR`/`TEXT` on PostgreSQL.

Verified by compiling the same models to both dialects:

```
PostgreSQL : id SERIAL,  created_at TIMESTAMP ... DEFAULT (NOW() AT TIME ZONE 'utc')
SQL Server : id INTEGER IDENTITY, created_at DATETIME2 ... DEFAULT SYSUTCDATETIME()
```

## Setup

1. Create a project at [supabase.com](https://supabase.com) (free tier is enough).
2. **Project Settings → Database → Connection string → URI**. Use the
   **Session pooler** URI (port 5432); the transaction pooler does not support
   prepared statements, which SQLAlchemy uses.
3. Put it in `.env` at the repo root:

   ```env
   DATABASE_URL=postgresql://postgres.xxxx:PASSWORD@aws-0-eu-central-1.pooler.supabase.com:5432/postgres
   JWT_SECRET_KEY=<a long random string>
   ```

4. Create the tables:

   ```bash
   .venv/bin/alembic upgrade head
   ```

   This also seeds the `super_admin`, `admin` and `agent` roles, which
   `scripts/create_super_admin.py` requires.

5. Create the first user:

   ```bash
   PYTHONPATH=. .venv/bin/python scripts/create_super_admin.py
   ```

6. Run everything on one port:

   ```bash
   PYTHONPATH=.:rag/src .venv/bin/uvicorn app.main:app --reload --port 8000
   ```

   * <http://127.0.0.1:8000/> — the portal, with **Knowledge Base** in the sidebar
   * <http://127.0.0.1:8000/docs> — API docs

Preview the SQL before applying anything:

```bash
.venv/bin/alembic upgrade head --sql
```

## ⚠️ The `.bak` file your colleague sent

`contact_center_db` is a **SQL Server backup**. PostgreSQL cannot read it —
the format is proprietary, and there is no converter.

The *schema* does not need it: `alembic upgrade head` builds every table from
the models. Only the **data** inside it is unreachable.

If that data matters, it has to pass through SQL Server once:

1. `docker run -e ACCEPT_EULA=Y -e MSSQL_SA_PASSWORD=<pw> -p 1433:1433 -d mcr.microsoft.com/mssql/server:2022-latest`
2. Restore the `.bak` into it.
3. Export the five tables to CSV.
4. Import the CSVs into Supabase (Table Editor → Import).

If it is only test data, skip all of that and seed fresh with
`create_super_admin.py`.

## Auth modes

`RAG_AUTH_MODE` in `rag/.env` controls where the knowledge base gets its
tenant:

| value | `company_id` from | needs the database |
|---|---|---|
| `dev` | `X-Company-Id` header | no |
| `jwt` | the signed-in user's row | yes |

Switch to `jwt` once Supabase is connected. A user with no company (a super
admin) is then refused rather than shown every tenant's documents.

## Keeping SQL Server instead

Leave `DATABASE_URL` empty and set `DB_SERVER`, `DB_NAME`, `DB_DRIVER`, plus
`DB_USER`/`DB_PASSWORD` for SQL authentication. You would still need
`msodbcsql18` installed locally; `pyodbc.drivers()` currently returns `[]` on
this machine.
