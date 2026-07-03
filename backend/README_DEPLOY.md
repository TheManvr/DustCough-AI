# DustCough AI Backend Deployment

Deploy this FastAPI service to Render.

## Render Settings

- Root Directory: `backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`

## Health Check

`GET /health`

Expected response:

```json
{ "status": "ok" }
```

## CORS

During testing, `main.py` allows all origins:

```python
allow_origins=["*"]
```

TODO: Restrict this to the production Vercel frontend domain after the final URL is known.

## Model File

`/predict-cough` requires `backend/model/cough_model.joblib`.

The project intentionally ignores model artifacts and dataset/audio files in Git. For production, provide the trained model through a trusted deployment artifact or generate it before deployment.

## API Endpoint

`POST /predict-cough`

Accepts uploaded audio with multipart form field:

```text
file
```

The frontend sends captured WAV audio to this endpoint.
