import pandas as pd
from scipy.signal import savgol_filter
from typing import List


class SGFilter:
    """
    Savitzky-Golay Filter for the American futures market gold prices dataset.
    """
    window_length: int = 3
    polyorder: int = 1
    features: List[str] = ['Close', 'High', 'Low', 'Open', 'Volume']

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for feature in self.features:
            df[feature] = savgol_filter(
                df[feature],
                window_length=self.window_length,
                polyorder=self.polyorder
            )
        return df
