from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def run_rf_model():

    project_root = Path(__file__).resolve().parents[1]

    df = pd.read_csv(
        project_root / "data" / "transformed" / "modeling_table.csv"
    )

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

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]

    roc = roc_auc_score(y_test, probs)

    print("RF ROC:", roc)

    output_dir = project_root / "data" / "model_outputs"

    fi = pd.DataFrame({
        "feature":features,
        "importance":model.feature_importances_
    })

    fi.to_csv(output_dir / "rf_feature_importance.csv", index=False)

    return model