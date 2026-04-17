# Water Main Break Prediction Project

## Project Overview

This project focuses on understanding and predicting daily water main break risk using historical repair work order data and weather conditions. The goal is to explore whether environmental factors such as freezing temperatures, precipitation, and temperature swings are associated with higher break activity.

The analysis combines NOAA daily weather observations with Montgomery County water main repair records. A full data pipeline was built to extract, clean, transform, and model the data in a structured way.

Two machine learning models were tested:

- Logistic Regression as a baseline model  
- Random Forest to capture more complex nonlinear relationships  

---

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/rmertus1-cmyk/inst737-final-project-rabbouni-mertus.git
cd inst737-final-project-rabbouni-mertus
pip install -r requirements.txt
Then run:
python main.py
---

inst737-final-project-rabbouni-mertus/
│
├── data/
│   ├── source/
│   ├── extracted/
│   ├── transformed/
│   ├── model_outputs/
│   └── visualizations/
│
├── etl/
│   ├── extract.py
│   ├── transform.py
│   └── load.py
│
├── analysis/
│   ├── model_logistic.py
│   └── model_rf.py
│
├── vis/
│   └── visualizations.py
│
├── logs/
│   └── pipeline.log
│
└── main.py

---

###  Pipeline Steps

```md
---

## Pipeline Steps
1. Extract
Reads raw weather and water main break data and saves copies to data/extracted/.
2. Transform
Cleans and aggregates weather data, creates freeze and freeze-thaw features, and builds a daily modeling dataset.
3. Model
Trains Logistic Regression and Random Forest models and evaluates performance.
4. Visualization
Generates plots for exploratory analysis and reporting.
5. Load
Validates that final outputs exist and confirms pipeline completion.

---

## Model Evaluation

Two models were evaluated:

- Logistic Regression  
  - ROC AUC ≈ 0.64  

- Random Forest  
  - ROC AUC ≈ 0.54  

Evaluation outputs are saved in:
`data/model_outputs/`

These include:
- Classification reports  
- ROC scores  
- Confusion matrices  
- Feature importance (Random Forest)

---

## Logging and Error Handling

The pipeline includes logging and error handling to improve robustness:

- Logging is implemented using Python’s `logging` module  
- Logs are saved to `logs/pipeline.log`  
- Each stage (extract, transform, model, visualize) includes try/except handling  
- Errors are logged without stopping the entire pipeline when possible  
