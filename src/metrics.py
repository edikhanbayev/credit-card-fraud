import pandas as pd
import numpy as np


from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc
)


def evaluate_model(
    y_true,
    probabilities,
    threshold=0.5
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true,
            probabilities
        )
    )

    return {
        "threshold": threshold,

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities
        ),

        "average_precision":
            average_precision_score(
                y_true,
                probabilities
            ),

        "pr_auc": auc(
            recall_curve,
            precision_curve
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
def find_best_threshold(
    y_true,
    probabilities,
    false_negative_cost,
    false_positive_cost
):
    thresholds = np.arange(
        0.001,
        1.000,
        0.001
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        business_cost = (
            fn * false_negative_cost
            +
            fp * false_positive_cost
        )

        rows.append(
            {
                "threshold": threshold,
                "business_cost": business_cost,
                "fp": fp,
                "fn": fn,
                "tp": tp,

                "precision":
                    precision_score(
                        y_true,
                        predictions,
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        y_true,
                        predictions,
                        zero_division=0
                    )
            }
        )

    result = pd.DataFrame(rows)

    best_row = result.loc[
        result[
            "business_cost"
        ].idxmin()
    ]

    return best_row, result