from pathlib import Path
import pandas as pd


def extract_data():
    """Read source files from the project and save raw copies into data/extracted."""

    project_root = Path(__file__).resolve().parents[1]

    source_dir = project_root / "data" / "source"
    extracted_dir = project_root / "data" / "extracted"

    extracted_dir.mkdir(parents=True, exist_ok=True)

    weather_file = source_dir / "weather.xlsx"
    breaks_file = source_dir / "water_main.csv"

    weather_df = pd.read_excel(weather_file)
    breaks_df = pd.read_csv(breaks_file)

    weather_df.to_csv(extracted_dir / "weather_raw.csv", index=False)
    breaks_df.to_csv(extracted_dir / "water_main_raw.csv", index=False)

    print("Raw datasets saved to data/extracted")

    return weather_df, breaks_df