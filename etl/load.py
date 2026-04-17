import logging
from pathlib import Path


def load_data():
    """
    Load stage: confirms final outputs exist and are ready for use.
    """

    logging.info("Load: starting load stage")

    try:
        project_root = Path(__file__).resolve().parents[1]

        transformed_dir = project_root / "data" / "transformed"
        model_outputs_dir = project_root / "data" / "model_outputs"

        modeling_file = transformed_dir / "modeling_table.csv"

        if not modeling_file.exists():
            raise FileNotFoundError("Modeling table not found")

        logging.info(f"Modeling table exists: {modeling_file}")

        logging.info(f"Model outputs directory: {model_outputs_dir}")

        logging.info("Load: completed successfully")

    except FileNotFoundError:
        logging.exception("Load: missing expected output file")
        raise

    except Exception:
        logging.exception("Load: unexpected error")
        raise
  