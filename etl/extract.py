from pathlib import Path
import pandas as pd
import logging


def extract_data():
    """
    Read source files and save raw copies into data/extracted.
    """

    logging.info("Extract: starting data extraction")

    try:
        project_root = Path(__file__).resolve().parents[1]

        source_dir = project_root / "data" / "source"
        extracted_dir = project_root / "data" / "extracted"

        extracted_dir.mkdir(parents=True, exist_ok=True)

        weather_file = source_dir / "weather.xlsx"
        breaks_file = source_dir / "water_main.csv"

        logging.info(f"Reading weather file: {weather_file}")
        weather_df = pd.read_excel(weather_file)

        logging.info(f"Reading breaks file: {breaks_file}")
        breaks_df = pd.read_csv(breaks_file)

        weather_output = extracted_dir / "weather_raw.csv"
        breaks_output = extracted_dir / "water_main_raw.csv"

        weather_df.to_csv(weather_output, index=False)
        breaks_df.to_csv(breaks_output, index=False)

        logging.info(f"Saved weather data to: {weather_output}")
        logging.info(f"Saved breaks data to: {breaks_output}")
        logging.info("Extract: completed successfully")

        return weather_df, breaks_df

    except FileNotFoundError:
        logging.exception("Extract: file not found error")
        raise

    except Exception:
        logging.exception("Extract: unexpected error")
        raise