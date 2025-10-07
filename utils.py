import pandas as pd

def basic_profile(df: pd.DataFrame) -> dict:
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "nulls": df.isna().sum().to_dict(),
        "sample": df.head(5).to_dict(orient="records"),
        "describe_num": df.describe(include='number').to_dict(),
    }
