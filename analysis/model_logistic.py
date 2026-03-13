from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def run_logistic_model():

    project_root = Path(__file__).resolve().parents[1]

    data_path = project_root / "data" / "transformed" / "modeling_table.csv"

    output_dir = project_root / "data" / "model_outputs"
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)

    df = df.dropna()

    features = [
        "tmin",
        "tmax",
        "prcp",
        "snow",
        "freeze_flag",
        "freeze_thaw_flag"
    ]

    X = df[features]
    y = df["break_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    report = classification_report(y_test, preds, output_dict=True)

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(output_dir / "logistic_metrics.csv")

    roc = roc_auc_score(y_test, probs)

    roc_df = pd.DataFrame({"roc_auc":[roc]})

    roc_df.to_csv(output_dir / "logistic_roc.csv", index=False)

    print("Saved model outputs")

    return model