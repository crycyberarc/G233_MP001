"""Train and register the initial delay model from the original training period."""

import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, TARGET, clean_train, load_config, update_findings


def make_pipeline(candidate, seed):
    settings = dict(candidate)
    name = settings.pop("name")
    models = {
        "ridge": Ridge,
        "random_forest": RandomForestRegressor,
        "histogram_gradient_boosting": HistGradientBoostingRegressor,
    }
    if name != "ridge":
        settings["random_state"] = seed
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), NUMERIC),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline([("preprocess", preprocessing), ("model", models[name](**settings))])


def main():
    config = load_config()
    paths = config["data"]
    settings = config["training"]
    original = Path(paths["train"])
    fingerprint = hashlib.sha256(original.read_bytes()).hexdigest()
    data = clean_train(pd.read_csv(original)).sort_values("date", kind="stable")
    split_at = int(len(data) * (1 - settings["validation_fraction"]))
    cutoff = data["date"].iloc[split_at]
    fit_rows = data.loc[data["date"] < cutoff]
    valid_rows = data.loc[data["date"] >= cutoff]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    name = settings["registered_model"]
    if client.search_model_versions(f"name = '{name}'"):
        raise RuntimeError("The registered v1 already exists. Use a clean workspace for a reproduction.")
    if client.get_experiment_by_name(settings["experiment"]) is None:
        client.create_experiment(settings["experiment"], artifact_location=settings["artifact_location"])
    mlflow.set_experiment(settings["experiment"])
    results = []
    for candidate in settings["candidates"]:
        with mlflow.start_run(run_name=candidate["name"]) as run:
            pipeline = make_pipeline(candidate, settings["seed"])
            pipeline.fit(fit_rows[FEATURES], fit_rows[TARGET])
            mae = float(mean_absolute_error(valid_rows[TARGET], pipeline.predict(valid_rows[FEATURES])))
            mlflow.log_params({
                **candidate,
                "data_sha256": fingerprint,
                "validation_start": cutoff.date().isoformat(),
                "fit_rows": len(fit_rows),
                "validation_rows": len(valid_rows),
                "random_seed": settings["seed"],
            })
            mlflow.log_metric("mae_validation", mae)
            results.append((mae, run.info.run_id, pipeline))
            print(f"{candidate['name']}: {mae:.4f} minutes ({run.info.run_id})")

    best_mae, run_id, pipeline = min(results, key=lambda result: result[0])
    champion = clone(pipeline).fit(data[FEATURES], data[TARGET])
    example = data[FEATURES].dropna().head(3)
    signature = infer_signature(example, champion.predict(example))
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("export_fit_rows", len(data))
        logged = mlflow.sklearn.log_model(
            champion,
            name="v1",
            serialization_format="cloudpickle",
            signature=signature,
        )
        version = mlflow.register_model(logged.model_uri, name)
        client.set_registered_model_alias(name, "champion", version.version)

    export = Path(settings["export"])
    export.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, export, compress=3)
    update_findings(paths["findings"], "tracking", {
        "experiment_name": settings["experiment"],
        "best_run_id": run_id,
        "registered_model_name": name,
        "champion_version": int(version.version),
    })
    print(f"{name}@champion = version {version.version}, validation MAE {best_mae:.4f}")


if __name__ == "__main__":
    main()
