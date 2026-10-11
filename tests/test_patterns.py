"""Basic tests for pattern detection."""

import numpy as np
import pandas as pd

from finsell.patterns import find_swing_highs, find_swing_lows


def test_swing_highs_detected():
    # Create a simple wave pattern
    x = np.sin(np.linspace(0, 4 * np.pi, 100))
    df = pd.DataFrame({"High": x, "Low": x, "Close": x})
    highs = find_swing_highs(df, order=5, col="High")
    assert highs.sum() > 0


def test_swing_lows_detected():
    x = np.sin(np.linspace(0, 4 * np.pi, 100))
    df = pd.DataFrame({"High": x, "Low": x, "Close": x})
    lows = find_swing_lows(df, order=5, col="Low")
    assert lows.sum() > 0
