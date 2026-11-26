import pandas as pd

from src.bus_delay.data import clean_train


def test_historical_repairs_preserve_targets_and_source():
    base = {
        "route_km": 12.0, "stops": 20, "passengers": 15, "hour": 8,
        "temp_c": 28.0, "driver_exp_yr": 72.0, "weather": " Cle. ",
        "depot_zone": "East", "delay_min": 18.0,
    }
    rows = [
        {**base, "date": date.date().isoformat()}
        for date in pd.date_range("2026-02-23", periods=7)
    ]
    rows.append({**base, "date": "2026-03-02", "passengers": 40})
    original = pd.DataFrame(rows)
    cleaned = clean_train(original)

    assert cleaned["passengers"].isna().tolist() == [True] * 7 + [False]
    assert cleaned["weather"].eq("clear").all()
    assert cleaned["driver_exp_yr"].eq(6.0).all()
    assert cleaned["delay_min"].eq(18.0).all()
    assert original["driver_exp_yr"].eq(72.0).all()
    assert original["passengers"].notna().all()
