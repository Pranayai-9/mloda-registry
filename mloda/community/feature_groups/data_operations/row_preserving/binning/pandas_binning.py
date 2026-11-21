import pandas as pd
import numpy as np
from mloda.community.feature_groups.data_operations.row_preserving.binning.base import Binning
from mloda.community.feature_groups.data_operations.row_preserving.arithmetic.pandas_numeric_source import pandas_non_numeric_descriptor

class PandasBinning(Binning):
    def _validate_source(self, source: pd.Series):
        if not np.issubdtype(source.dtype, np.number) or np.iscomplexobj(source):
            raise ValueError("Binning needs an integer or floating source column")

    def bin(self, source: pd.Series, bins: int) -> pd.Series:
        self._validate_source(source)
        return pd.cut(source, bins, labels=False)

    def qbin(self, source: pd.Series, bins: int) -> pd.Series:
        self._validate_source(source)
        return pd.qcut(source, bins, labels=False, duplicates='drop')
