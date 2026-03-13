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
