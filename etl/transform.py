from pathlib import Path
import pandas as pd


def transform_data():

    project_root = Path(__file__).resolve().parents[1]

    extracted_dir = project_root / "data" / "extracted"
    transformed_dir = project_root / "data" / "transformed"

    transformed_dir.mkdir(parents=True, exist_ok=True)

    weather = pd.read_csv(extracted_dir / "weather_raw.csv")

    weather.columns = weather.columns.str.lower()

    weather["date"] = pd.to_datetime(weather["date"])

    daily_weather = (
        weather
        .groupby("date")
        .agg(
            tmin=("tmin", "min"),
            tmax=("tmax", "max"),
            prcp=("prcp", "mean"),
            snow=("snow", "mean")
        )
        .reset_index()
        .sort_values("date")
    )

    daily_weather["freeze_flag"] = (daily_weather["tmin"] <= 32).astype(int)

    daily_weather["freeze_thaw_flag"] = (
        (daily_weather["tmin"] <= 32) &
        (daily_weather["tmax"] > 32)
    ).astype(int)

    # WATER MAIN BREAKS
    breaks = pd.read_csv(extracted_dir / "water_main_raw.csv")

    breaks.columns = breaks.columns.str.lower()

    print("Break dataset columns:")
    print(breaks.columns)

    # GUESS repair date column
    date_col = None

    for c in breaks.columns:
        if "date" in c:
            date_col = c
            break

    if date_col is None:
        raise ValueError("No date column found in break dataset")

    breaks["date"] = pd.to_datetime(breaks[date_col])

    daily_breaks = (
        breaks
        .groupby("date")
        .size()
        .reset_index(name="break_count")
    )

    daily_breaks["break_flag"] = (daily_breaks["break_count"] > 0).astype(int)

    print("Daily breaks preview:")
    print(daily_breaks.head())

    modeling_table = daily_weather.merge(
        daily_breaks,
        on="date",
        how="left"
    )

    modeling_table["break_count"] = modeling_table["break_count"].fillna(0)

    modeling_table["break_flag"] = modeling_table["break_flag"].fillna(0)

    modeling_table.to_csv(
        transformed_dir / "modeling_table.csv",
        index=False
    )

    print("Saved modeling_table.csv")

    return modeling_table