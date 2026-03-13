from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def create_visualizations():

    project_root = Path(__file__).resolve().parents[1]

    df = pd.read_csv(
        project_root / "data" / "transformed" / "modeling_table.csv"
    )

    vis_dir = project_root / "data" / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # 1 break trend
    trend = df.groupby("date")["break_count"].sum()

    plt.figure(figsize=(12,5))
    trend.plot()
    plt.title("Daily Water Main Break Counts")
    plt.savefig(vis_dir / "break_trend.png")
    plt.close()

    # 2 freeze vs breaks
    freeze_rate = df.groupby("freeze_flag")["break_flag"].mean()

    plt.figure()
    freeze_rate.plot(kind="bar")
    plt.title("Break Probability: Freeze vs No Freeze")
    plt.savefig(vis_dir / "freeze_vs_break.png")
    plt.close()

    # 3 precipitation vs breaks
    plt.figure()
    plt.scatter(df["prcp"], df["break_count"])
    plt.title("Precipitation vs Break Count")
    plt.savefig(vis_dir / "prcp_vs_break.png")
    plt.close()

    print("Saved visualizations")