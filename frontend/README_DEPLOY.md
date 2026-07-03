# DustCough AI Frontend Deployment

Deploy this React + Vite app to Vercel.

## Vercel Settings

- Framework Preset: `Vite`
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`

## Environment Variables

Set this in Vercel:

```text
VITE_API_BASE_URL=https://YOUR_RENDER_BACKEND_URL
```

Example:

```text
VITE_API_BASE_URL=https://dustcough-ai-backend.onrender.com
```

Do not put secret keys in `VITE_` variables. Vite exposes them to the browser.

## Microphone Requirement

DustCough AI uses `navigator.mediaDevices.getUserMedia`, so production must run on HTTPS. Vercel provides HTTPS for deployed frontend URLs.

## Backend Calls

The app reads the backend base URL from:

```js
import.meta.env.VITE_API_BASE_URL
```

Local fallback:

```text
http://localhost:8000
```
