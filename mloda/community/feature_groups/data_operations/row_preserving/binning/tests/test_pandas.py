import pandas as pd
import pytest
from mloda.community.feature_groups.data_operations.row_preserving.binning.pandas_binning import PandasBinning

def test_pandas_binning_rejects_complex():
    binning = PandasBinning()
    source = pd.Series([1 + 1j, 2 + 0j])
    with pytest.raises(ValueError, match="Binning needs an integer or floating source column"):
        binning.bin(source, 2)
    with pytest.raises(ValueError, match="Binning needs an integer or floating source column"):
        binning.qbin(source, 2)
