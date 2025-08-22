import pandas as pd
from src.data.validate_schema import coerce_and_validate

def test_schema_pass():
    df = pd.DataFrame({
        "Age":[30], "Gender":[1], "BMI":[24.5], "Smoking":[0],
        "GeneticRisk":[1], "PhysicalActivity":[3.0], "AlcoholIntake":[1.0],
        "CancerHistory":[0], "Diagnosis":[0]
    })
    out = coerce_and_validate(df, [
        "Age","Gender","BMI","Smoking","GeneticRisk",
        "PhysicalActivity","AlcoholIntake","CancerHistory"
    ], "Diagnosis")
    assert len(out)==1
