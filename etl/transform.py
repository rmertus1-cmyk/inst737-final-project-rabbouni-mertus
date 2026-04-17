from pathlib import Path
import pandas as pd
import logging


def transform_data():
    """
    Transform extracted weather and water main break data into a modeling table.
    """

    logging.info("Transform: starting data transformation")

    try:
        project_root = Path(__file__).resolve().parents[1]

        extracted_dir = project_root / "data" / "extracted"
        transformed_dir = project_root / "data" / "transformed"

        transformed_dir.mkdir(parents=True, exist_ok=True)

        weather_file = extracted_dir / "weather_raw.csv"
        breaks_file = extracted_dir / "water_main_raw.csv"

        logging.info(f"Reading transformed input weather file: {weather_file}")
        weather = pd.read_csv(weather_file)

        weather.columns = weather.columns.str.lower()
        weather["date"] = pd.to_datetime(weather["date"])

        logging.info("Creating daily weather aggregates")
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

        logging.info(f"Reading transformed input breaks file: {breaks_file}")
        breaks = pd.read_csv(breaks_file)

        breaks.columns = breaks.columns.str.lower()
        logging.info(f"Break dataset columns: {list(breaks.columns)}")

        date_col = None
        for c in breaks.columns:
            if "date" in c:
                date_col = c
                break

        if date_col is None:
            raise ValueError("No date column found in break dataset")

        logging.info(f"Using break date column: {date_col}")
        breaks["date"] = pd.to_datetime(breaks[date_col])

        logging.info("Creating daily break counts")
        daily_breaks = (
            breaks
            .groupby("date")
            .size()
            .reset_index(name="break_count")
        )

        daily_breaks["break_flag"] = (daily_breaks["break_count"] > 0).astype(int)
        logging.info(f"Daily breaks preview:\n{daily_breaks.head().to_string(index=False)}")

        logging.info("Merging weather and break data into modeling table")
        modeling_table = daily_weather.merge(
            daily_breaks,
            on="date",
            how="left"
        )

        modeling_table["break_count"] = modeling_table["break_count"].fillna(0)
        modeling_table["break_flag"] = modeling_table["break_flag"].fillna(0)

        output_file = transformed_dir / "modeling_table.csv"
        modeling_table.to_csv(output_file, index=False)

        logging.info(f"Saved modeling table to: {output_file}")
        logging.info("Transform: completed successfully")

        return modeling_table

    except FileNotFoundError:
        logging.exception("Transform: file not found error")
        raise

    except KeyError:
        logging.exception("Transform: missing expected column in dataset")
        raise

    except ValueError:
        logging.exception("Transform: value error during transformation")
        raise

    except Exception:
        logging.exception("Transform: unexpected error")
        raise