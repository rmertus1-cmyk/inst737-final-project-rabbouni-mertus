"""
Main pipeline runner for the INST737 final project.
Runs extraction, transformation, modeling, and visualization stages
with logging and basic error handling.
"""

import logging
from pathlib import Path

from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model_logistic import run_logistic_model
from analysis.model_rf import run_rf_model
from vis.visualizations import create_visualizations


def setup_logging():
    """
    Set up logging to both console and file.
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    return log_file


def main():
    log_file = setup_logging()
    logging.info("Pipeline started")
    logging.info(f"Log file: {log_file}")

    try:
        logging.info("Starting extract stage")
        extract_data()
        logging.info("Extract stage completed")
    except Exception as e:
        logging.exception(f"Extract stage failed: {e}")
        return

    try:
        logging.info("Starting transform stage")
        transform_data()
        logging.info("Transform stage completed")
    except Exception as e:
        logging.exception(f"Transform stage failed: {e}")
        return

    try:
        logging.info("Starting logistic model stage")
        run_logistic_model()
        logging.info("Logistic model stage completed")
    except Exception as e:
        logging.exception(f"Logistic model stage failed: {e}")

    try:
        logging.info("Starting random forest model stage")
        run_rf_model()
        logging.info("Random forest model stage completed")
    except Exception as e:
        logging.exception(f"Random forest model stage failed: {e}")

    try:
        logging.info("Starting visualization stage")
        create_visualizations()
        logging.info("Visualization stage completed")
    except Exception as e:
        logging.exception(f"Visualization stage failed: {e}")

    try:
        logging.info("Starting load stage")
        load_data()
        logging.info("Load stage completed")
    except Exception as e:
        logging.exception(f"Load stage failed: {e}")

    logging.info("Pipeline finished")


if __name__ == "__main__":
    main()