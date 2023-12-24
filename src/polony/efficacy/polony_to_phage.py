from typing import Tuple

import numpy as np
import pandas as pd


def pol_to_phage(
    X: float, virus_type: str, alpha: float = 0.05, n: int = 10000
) -> Tuple[float, float]:
    """Convert polony count to phage abundance using bootstrap method.

    Args:
    X (float): Polony count.
    virus_type (str): Type of virus. Can be T4, T7 or T7c.
    alpha (float): Confidence level for the bootstrap interval.
        Defaulyts to 0.05
    n (int): Number of bootstrap samples.
        Defaults to 10000.

    Returns:
    Tuple[float, float]: A tuple containing the mean phage abundance and the half-width
        of the confidence interval.
    """
    if virus_type not in ["T4", "T7", "T7c"]:
        raise ValueError("Wrong type of virus, should be T4, T7 or T7c")
    # Predefined slopes for calculation
    if virus_type == "T4":
        slope = np.array(
            [2.3476621, 1.697986428, 2.284610116, 2.406920764, 2.241928335, 4.211469763]
        )
    elif virus_type == "T7":
        slope = np.array(
            [1.612903226, 2.222222222, 2.173913043, 3.225806452, 5.555555556]
        )
    elif virus_type == "T7c":
        raise ValueError("This option will be added in 2024 year")
    product_bootstrap = np.zeros(n)

    for i in range(n):
        # Sampling the slopes with replacement
        slope_bootstrap = np.random.choice(slope, size=len(slope), replace=True)
        # Calculating bootstrap sample
        product_bootstrap[i] = np.mean(slope_bootstrap) * X

    # Calculating the mean and confidence interval
    phages_mean = np.mean(product_bootstrap)
    phages_ci = np.quantile(product_bootstrap, [alpha / 2, 1 - alpha / 2])
    return phages_mean, (phages_ci[1] - phages_ci[0]) / 2
