from etl.extract import extract_data
from etl.transform import transform_data
from etl.load import load_data
from analysis.model_1 import run_model_1
from vis.visualizations import create_visualizations

def main():
    raw = extract_data()
    transformed = transform_data(raw)
    load_data(transformed)
    run_model_1(transformed)
    create_visualizations(transformed)

if __name__ == "__main__":
    main()