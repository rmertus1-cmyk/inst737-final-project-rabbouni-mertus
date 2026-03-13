"""
Runs extraction, transformation, modeling, and visualization stages.
"""

from etl.extract import extract_data
from etl.transform import transform_data
from analysis.model_logistic import run_logistic_model
from analysis.model_rf import run_rf_model
from vis.visualizations import create_visualizations


def main():
    extract_data()
    transform_data()
    run_logistic_model()
    run_rf_model()
    create_visualizations()


if __name__ == "__main__":
    main()