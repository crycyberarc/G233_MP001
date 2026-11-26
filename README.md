# City Bus Delay (Lucknow)

SCSE3040 MLOps mini project · Group **G233** · Problem **MP001**

Enrolment numbers: S24CSEU1517, S24CSEU1523, S24CSEU1527, S24CSEU1516.

The model estimates how many minutes late a bus will reach its final stop, using information available when the trip starts. Validation uses mean absolute error in minutes.

## Local setup

Use Python 3.12 and install the pinned dependencies with `python -m pip install -r requirements.txt`. Run `python -m pytest -q` from the repository root. Tests use synthetic inputs.

Extract `train.csv` and `later.csv` from the original ZIP into `data/raw/`. Keep the original CSV bytes intact. DVC tracks the original train and later files through their committed pointers. The holdout features are at `data/raw/holdout_features.csv`.

Run `python -m src.bus_delay.data` to audit the raw data. Run `python -m src.bus_delay.train` with a fresh local tracking database to reproduce v1 from the original train data. The tracked `mlflow.db` records the successful v1 runs.

## Serve predictions

The service loads the v1 champion from `models/model.joblib`. Start the API with `python -m uvicorn app.main:app --port 8000`. The interactive API is at `/docs`, the status endpoint is `GET /health`, and the prediction endpoint is `POST /predict`.

Build and run the deployment image with `docker build -t mp001-api:1.0 .` and `docker run --rm -p 8000:8000 mp001-api:1.0`. Kubernetes manifests are in `k8s/`.

Each teammate uses their own GitHub account for commits and pull requests.
