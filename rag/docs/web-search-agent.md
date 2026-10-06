# Web-search agent

The Search page delegates web searches to an n8n workflow. Configure the
webhook URL in `rag/.env`:

```dotenv
WEB_SEARCH_N8N_WEBHOOK_URL=https://n8n.example/webhook/web-search
WEB_SEARCH_TIMEOUT_SECONDS=120
```

The backend sends:

```json
{
  "query": "the user's search text",
  "company_id": 123
}
```

The workflow can return an object with `answer` (or `response`, `text`, or
`output`) and optional `sources` or `results`. A source can be an object such
as `{"title": "...", "url": "https://..."}` or a string. The backend
normalizes these into the response consumed by the UI.

The browser never receives the webhook URL. Requests use the signed-in
company's tenant scope, and a super admin must select a company before
searching.
