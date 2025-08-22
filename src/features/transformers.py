from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class PassthroughFrame(BaseEstimator, TransformerMixin):
    """Wrap numpy array into a DataFrame with provided column names."""
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns)
