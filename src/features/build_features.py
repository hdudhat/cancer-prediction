from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def make_preprocess(scale_columns):
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), scale_columns)],
        remainder="passthrough"
    )
