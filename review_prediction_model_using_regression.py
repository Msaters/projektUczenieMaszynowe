import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from config import *


def all_results(embeddings_filepath, csv_filepath, target_col="points", title_change="") -> None:

    df = load_embeddings_and_data(embeddings_filepath, csv_filepath)
    nan_stats = df.isnull().mean(axis=0)
    columns_to_drop = nan_stats.index[nan_stats > 0]

    train_df, test_df = train_test_split(df, test_size=0.2)

    use_columns = train_df.select_dtypes(include="number").columns
    use_columns = use_columns.drop(target_col, errors="ignore")
    use_columns = use_columns.drop(columns_to_drop, errors="ignore")

    X = train_df[use_columns]
    y = train_df[target_col]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    print("Intercept: ", model.intercept_)
    print("Coefficients: ", model.coef_)

    X_test = test_df[use_columns]
    y_test = test_df[target_col]
    y_pred = model.predict(X_test)

    rsc = model.score(X_test, y_test)
    mse = skm.mean_squared_error(y_test, y_pred)
    mape = skm.mean_absolute_percentage_error(y_test, y_pred)
    mae = skm.mean_absolute_error(y_test, y_pred)

    all_results = [dict(variant="1-basic", MSE=mse, MAE=mae, MAPE=mape, Rsq=rsc)]
    print(pd.DataFrame(all_results))

    pred_vs_actual = pd.DataFrame({
        "points": y_test,
        "predicted_points": y_pred
    })

    forced_points = list(range(80, 101))

    data = [
        pred_vs_actual.loc[pred_vs_actual["points"] == p, "predicted_points"]
        if (pred_vs_actual["points"] == p).any()
        else pd.Series(dtype=float)
        for p in forced_points
    ]
    plt.figure(figsize=(12, 6))
    plt.boxplot(
        data,
        positions=range(1, len(forced_points) + 1),
        widths=0.6,
        showfliers=False
    )

    x_pos = {p: i + 1 for i, p in enumerate(forced_points)}
    xs = [x_pos[p] for p in forced_points]
    ys = forced_points
    plt.scatter(xs, ys, color="red", zorder=3, label="y = x (80–100)")

    plt.xticks(
        ticks=range(1, len(forced_points) + 1),
        labels=forced_points, # type: ignore
        rotation=90
    )
    plt.yticks(forced_points)
    plt.xlabel("Actual points")
    plt.ylabel("Predicted points")
    plt.title(f"Predicted points distribution for each actual points value {title_change}")
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plot_filename = f"predicted_points_distribution_min_occ_{MIN_WORD_OCCURENCE}.png"
    plot_path = os.path.join("plots", plot_filename)
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    all_results(EMBEDEDINGS_FILEPATH_tf_idf_monograms, CSV_FILEPATH_UNCHANGED_DATA)
