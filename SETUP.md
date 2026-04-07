# marketforge-backend — Files to Copy from marketforge-ai

Copy these files/folders from `marketforge-ai` into this repo:

## Source code (copy entire folders)
- `src/` — entire marketforge package
- `api/` — FastAPI app
- `scripts/bootstrap.py` — DB schema initialiser
- `worker.py` — APScheduler pipeline worker

## Keep in this repo (already here)
- `requirements.txt`
- `Dockerfile`
- `railway.toml`
- `pyproject.toml`
- `.gitignore`

## Do NOT copy (showcase-only, not needed for production)
- `airflow/` — replaced by worker.py APScheduler
- `mlflow/` — MLflow tracking not needed in stripped version
- `notebooks/` — development only
- `tests/` — run locally, not deployed
- `docker-compose.yml` — Railway provides postgres+redis
- `dashboard/` — frontend is separate repo

## After copying, install and test locally:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/bootstrap.py
uvicorn api.main:app --reload
```
