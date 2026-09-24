import numpy as np
import pandas as pd


def build_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    result = df.copy()

    result["LogAmount"] = np.log1p(
        result["Amount"].clip(
            lower=0
        )
    )

    result["TimeHours"] = (
        result["Time"] / 3600.0
    )

    result = result.drop(
        columns=["Time"]
    )

    return result