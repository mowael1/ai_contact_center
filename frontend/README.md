Build the frontend for my AI Contact Center project using ONLY:

* HTML5
* CSS3
* Vanilla JavaScript

Do NOT use:

* React
* Next.js
* Vue
* Angular
* Vite
* npm
* Node.js packages
* Bootstrap
* Tailwind
* Axios
* Any external frontend framework

The frontend must communicate with my existing FastAPI backend using the native browser Fetch API.

## Existing Backend

Base API path:

`/api/v1`

Currently implemented endpoints:

Authentication:

* `POST /api/v1/auth/login`
* `GET /api/v1/users/me`

Companies:

* `GET /api/v1/companies/`
* `GET /api/v1/companies/{company_id}`
* `POST /api/v1/companies/`
* `PATCH /api/v1/companies/{company_id}`
* `PATCH /api/v1/companies/{company_id}/status`

Do NOT invent backend endpoints.

Do NOT implement fake API calls for features that do not have backend endpoints yet.

## Authentication

The login page must call:

`POST /api/v1/auth/login`

After receiving the access token:

1. Store the access token.
2. Send authenticated API requests using:

`Authorization: Bearer <access_token>`

3. Call:

`GET /api/v1/users/me`

4. Determine the user's actual role from the API response.

5. Redirect based on role:

* SUPER_ADMIN → `/super-admin/dashboard.html`
* ADMIN → `/admin/dashboard.html`
* AGENT → `/agent/dashboard.html`

Do not hardcode a user's role.

## Role Protection

Create reusable JavaScript authentication and authorization logic.

Every protected page must:

1. Verify that a token exists.
2. Call `/api/v1/users/me`.
3. Verify that the logged-in user has permission to access the page.
4. Redirect unauthorized users to `/403.html`.
5. Redirect unauthenticated users to the login page.

Remember that frontend authorization is only for UX. Backend authorization remains authoritative.

## Shared API Layer

Create:

`js/api.js`

It should contain a reusable API request function that:

* Uses Fetch API.
* Automatically adds the JWT Bearer token.
* Automatically handles JSON.
* Handles 401 by clearing the token and redirecting to login.
* Handles 403 by redirecting to the 403 page.
* Throws useful errors using FastAPI's `detail` response.

Do not repeat Fetch and Authorization logic in every page.

## Required Folder Structure

Create:

frontend/

* index.html
* 403.html

super-admin/

* dashboard.html
* companies.html
* company-details.html
* company-form.html

admin/

* dashboard.html

agent/

* dashboard.html

profile/

* profile.html

css/

* style.css

js/

* config.js
* api.js
* auth.js
* guard.js
* layout.js
* login.js
* profile.js

js/super-admin/

* dashboard.js
* companies.js
* company-details.js
* company-form.js

## Login Page

Create a professional login interface containing:

* AI Contact Center logo/title
* Email
* Password
* Login button
* Loading state
* Error message area

After successful login, retrieve the current user and redirect according to their role.

## Super Admin Dashboard

Use REAL backend data.

Load:

`GET /api/v1/companies/`

Display:

* Logged-in user's name
* Role
* Total number of companies
* Number of active companies
* Number of inactive companies
* Recent companies table

Do not display fake statistics for calls, customers, tickets, FCR, etc.

Those backend APIs do not exist yet.

## Companies Page

Create a responsive companies table.

Columns:

* Name
* Email
* Phone
* Status
* Actions

Actions:

* View
* Edit
* Activate / Deactivate

Add:

* Search field
* Add Company button
* Loading state
* Empty state
* Error state

Load data from:

`GET /api/v1/companies/`

Search can currently be implemented client-side.

## Company Details

Use query parameter:

`company-details.html?id={company_id}`

Load:

`GET /api/v1/companies/{company_id}`

Display all fields returned by the real API.

Include:

* Edit button
* Activate / Deactivate button
* Back to Companies button

Do not assume fields that are not returned by the API.

## Create / Edit Company

Use the same page:

`company-form.html`

Create mode:

`company-form.html`

Edit mode:

`company-form.html?id={company_id}`

In create mode call:

`POST /api/v1/companies/`

In edit mode:

1. Load the company using GET.
2. Populate the form.
3. Update using:

`PATCH /api/v1/companies/{company_id}`

Use the exact request schemas expected by the FastAPI backend.

Do not send fields that the backend schema does not accept.

Do NOT add "Create First Admin" to the company form yet because that onboarding API is not currently available.

## Company Status

Use:

`PATCH /api/v1/companies/{company_id}/status`

Provide a confirmation modal before changing status.

Refresh the UI after success.

## Admin Dashboard

For now, use only data available from:

`GET /api/v1/users/me`

Show:

* Welcome message
* Full name
* Role
* Company information if returned

Do not create fake customer/ticket/call statistics.

Create a clean placeholder area for future company management modules.

## Agent Dashboard

For now, use only:

`GET /api/v1/users/me`

Show:

* Welcome message
* Full name
* Role
* Company information if available

Do not create fake ticket, call, customer, or escalation data.

Leave a professional placeholder area for future agent workspace modules.

## Profile

Load current user information from:

`GET /api/v1/users/me`

Display the actual fields returned by the API.

Include Logout.

## Layout

Build a reusable layout system using JavaScript.

Header:

* Application name
* Current user
* Role
* Logout

Sidebar content must depend on role.

SUPER_ADMIN:

* Dashboard
* Companies
* Profile
* Logout

ADMIN:

* Dashboard
* Profile
* Logout

AGENT:

* Dashboard
* Profile
* Logout

Do not show inaccessible navigation items.

## UI Style

Create a modern SaaS dashboard design using only CSS.

Requirements:

* Dark sidebar
* Light main background
* White cards
* Modern typography
* Responsive layout
* Status badges
* Buttons
* Tables
* Forms
* Modal
* Loading indicators
* Empty states
* Error messages

Do not use external CSS frameworks.

Use CSS variables for common colors and spacing.

## JavaScript Standards

Use:

* `async / await`
* `fetch`
* `URLSearchParams`
* reusable functions
* ES modules if appropriate

Avoid global duplicated code.

Do not hardcode:

* users
* companies
* roles
* IDs
* statistics

Everything that already has an API must come from the backend.

## Important

Before connecting a page to an API, inspect the actual FastAPI request and response schema in the existing backend project.

Do not guess property names.

For example, if `/users/me` returns a role object instead of a role string, adapt the frontend to the actual response.

The existing backend is the source of truth.

The frontend must adapt to the backend, not change the backend just to fit assumptions in the frontend.

Build only the currently functional frontend first.

Future modules such as:

* Customers
* Tickets
* Follow-ups
* AI Calls
* Campaigns
* Knowledge Base
* Reports
* Integrations

will be added later when their backend endpoints are implemented.
