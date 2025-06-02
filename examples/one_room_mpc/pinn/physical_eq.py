import numpy as np
import pandas as pd

def T_pred(T):
    """
    Use historical data of output variable (T)
    Calculate last measured temperature difference and multiply with empirical chosen parameter
    """
    parameter = 1

    return parameter * T