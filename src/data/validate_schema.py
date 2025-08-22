import pandas as pd

EXPECTED_RANGES = {
    "Age": (20, 80),
    "Gender": (0, 1),
    "BMI": (15.0, 40.0),
    "Smoking": (0, 1),
    "GeneticRisk": (0, 2),
    "PhysicalActivity": (0.0, 10.0),
    "AlcoholIntake": (0.0, 5.0),
    "CancerHistory": (0, 1),
    "Diagnosis": (0, 1)
}

def coerce_and_validate(df: pd.DataFrame, features, target):
    # types
    int_cols = ["Age","Gender","Smoking","GeneticRisk","CancerHistory",target]
    float_cols = ["BMI","PhysicalActivity","AlcoholIntake"]

    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # drop NA
    df = df.dropna(subset=features + [target]).copy()

    # ranges
    for c,(lo,hi) in EXPECTED_RANGES.items():
        df = df[df[c].between(lo,hi)]

    for c in int_cols:
        df[c] = df[c].astype(int)

    df = df.drop_duplicates().reset_index(drop=True)
    return df
