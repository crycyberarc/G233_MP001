"""Data checks and historical repairs for Group G233."""

import json
from pathlib import Path

import pandas as pd
import yaml

NUMERIC = ["route_km", "stops", "passengers", "hour", "temp_c", "driver_exp_yr"]
CATEGORICAL = ["weather", "depot_zone"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "delay_min"

# The raw data audit reports the evidence for these G233-specific repairs.
EXPERIENCE_ZONE = "East"
MONTHS_PER_YEAR = 12
STUCK_READING = 15
MIN_STUCK_DAYS = 7


def load_config(path="params.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def passenger_sensor_runs(frame):
    """Return consecutive dates whose passenger readings are all stuck at 15."""
    daily = frame.groupby(pd.to_datetime(frame["date"]))["passengers"].agg(["min", "max"])
    stuck_dates = daily.index[daily["min"].eq(STUCK_READING) & daily["max"].eq(STUCK_READING)]
    dates = pd.Series(stuck_dates)
    groups = dates.diff().ne(pd.Timedelta(days=1)).cumsum()
    return [
        (run.iloc[0], run.iloc[-1])
        for _, run in dates.groupby(groups)
        if len(run) >= MIN_STUCK_DAYS
    ]


def normalize_weather(series):
    return series.str.strip().str.lower().replace({"cle.": "clear", "rai.": "rain", "fog.": "fog"})


def clean_train(frame):
    """Return clean features, the observation date, and the target."""
    cleaned = frame.loc[:, ["date", *FEATURES, TARGET]].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned["weather"] = normalize_weather(cleaned["weather"])
    in_east = cleaned["depot_zone"].eq(EXPERIENCE_ZONE)
    cleaned["driver_exp_yr"] = cleaned["driver_exp_yr"].astype(float)
    cleaned.loc[in_east, "driver_exp_yr"] /= MONTHS_PER_YEAR
    # Preserve rows and targets. The model imputes these historical missing
    # inputs using only the rows available to that model at fit time.
    cleaned["passengers"] = cleaned["passengers"].astype(float)
    for start, end in passenger_sensor_runs(frame):
        cleaned.loc[cleaned["date"].between(start, end), "passengers"] = float("nan")
    return cleaned


def audit_data(train, later, holdout):
    extras = sorted(set(train) - set(holdout) - {TARGET})
    duplicates = int(train.duplicated().sum())
    summary = train.groupby("depot_zone")["driver_exp_yr"].agg(["min", "median", "max"])
    east = train.loc[train["depot_zone"].eq(EXPERIENCE_ZONE), "driver_exp_yr"]
    other = summary.drop(index=EXPERIENCE_ZONE)
    unit_mixup = (
        east.min() == other["min"].min() * MONTHS_PER_YEAR
        and east.max() == other["max"].max() * MONTHS_PER_YEAR
        and bool(east.mod(MONTHS_PER_YEAR).eq(0).all())
    )
    runs = passenger_sensor_runs(train)
    start, end = runs[0] if len(runs) == 1 else (None, None)
    weather = normalize_weather(train["weather"])
    clean_weather = set(later["weather"].dropna())
    variants = bool(train["weather"].ne(weather).any()) and set(weather.dropna()) == clean_weather

    # Inspect the common sentinel candidates and the upper tail of the target.
    # High passenger frequency during the dated sensor outage is a stuck sensor.
    sentinel_hits = train[NUMERIC].isin([-1, -999, 999, 9999]).sum()
    target = train[TARGET].sort_values(ignore_index=True)
    upper = target.iloc[int(len(target) * 0.95):]
    max_upper_gap = float(target.diff().loc[upper.index].max())
    # A tenfold typo in this dataset would exceed the observed upper tail or
    # form a separated upper cluster. Keep the tail evidence in the audit output.
    print("Train-only fields excluding target:", extras)
    print("Driver experience by depot:\n" + summary.to_string())
    print("Sentinel candidate hits:", sentinel_hits.to_dict())
    print("Passenger sensor runs:", [(str(a.date()), str(b.date())) for a, b in runs])
    print("Target upper tail (unique):", upper.unique().tolist(), "largest gap:", max_upper_gap)
    print("Weather counts:", train["weather"].value_counts().to_dict())
    print("Clean weather levels:", sorted(clean_weather))
    if extras or sentinel_hits.any() or duplicates or max_upper_gap > 5:
        raise ValueError("Unexpected anomalies for this group. Review the audit evidence.")
    if not unit_mixup or len(runs) != 1 or not variants:
        raise ValueError("Unexpected anomalies for this group. Review the audit evidence.")

    return {
        "leaking_column": {"present": False, "column": None},
        "unit_mixup": {
            "present": bool(unit_mixup),
            "column": "driver_exp_yr" if unit_mixup else None,
            "group_value": EXPERIENCE_ZONE if unit_mixup else None,
            "factor": MONTHS_PER_YEAR if unit_mixup else None,
        },
        "sentinel_value": {"present": False, "column": None, "value": None, "count": None},
        "duplicate_rows": {"present": duplicates > 0, "count": duplicates or None},
        "stuck_sensor": {
            "present": start is not None,
            "column": "passengers" if start is not None else None,
            "start_date": start.date().isoformat() if start is not None else None,
            "end_date": end.date().isoformat() if end is not None else None,
        },
        "target_typos": {"present": False, "count": None},
        "spelling_variants": {
            "present": variants,
            "column": "weather" if variants else None,
            "true_levels": len(clean_weather) if variants else None,
        },
    }


def update_findings(path, section, values):
    report = Path(path)
    findings = yaml.safe_load(report.read_text(encoding="utf-8"))
    findings[section] = values
    report.write_text(yaml.safe_dump(findings, sort_keys=False), encoding="utf-8")


def main():
    paths = load_config()["data"]
    train, later, holdout = (pd.read_csv(paths[key]) for key in ("train", "later", "holdout"))
    findings = audit_data(train, later, holdout)
    print(json.dumps(findings, indent=2))
    update_findings(paths["findings"], "data_issues", findings)


if __name__ == "__main__":
    main()
