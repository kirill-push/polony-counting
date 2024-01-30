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
        slope = np.array([...])
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


def dat_for_plot(
    df: pd.DataFrame, virus_type: str, alpha: float = 0.05, n: int = 10000
) -> pd.DataFrame:
    """Process data and apply the pol_to_phage function.

    Args:
    df (pd.DataFrame): Dataframe containing the raw data.
    virus_type (str): Type of virus. Can be T4, T7 or T7c.
    alpha (float): Confidence level for the bootstrap interval.
        Defaulyts to 0.05
    n (int): Number of bootstrap samples.
        Defaults to 10000.

    Returns:
    pd.DataFrame: Dataframe with calculated phage abundances and confidence intervals.
    """
    # Grouping data by 'slide ID' and calculating mean of 'FAGE ABUNDENCE [fage mL-1]'
    df_grouped = (
        df.groupby("slide ID").agg({"FAGE ABUNDENCE [fage mL-1]": "mean"}).reset_index()
    )

    # Applying pol_to_phage function to each group
    # and extracting phages and confidence interval
    df_grouped["phages"], df_grouped["ci"] = zip(
        *df_grouped["FAGE ABUNDENCE [fage mL-1]"].apply(
            lambda x: pol_to_phage(x, virus_type=virus_type, alpha=alpha, n=n)
        )
    )

    # Calculating upper and lower confidence intervals
    df_grouped["upper_ci"] = df_grouped["phages"] + df_grouped["ci"]
    df_grouped["lower_ci"] = df_grouped["phages"] - df_grouped["ci"]
    return df_grouped[["slide ID", "phages", "upper_ci", "lower_ci"]]


def T4_efficacy_to_csv(
    input_path: str, output_path: str, alpha: float = 0.05, n: int = 10000
) -> None:
    """Function to calculate T4 phages from input file and write to output file results.

    Args:
        input_path (str): Path to csv with polony countings.
        output_path (str): Path to save results in csv file.
        alpha (float, optional): Confidence level for the bootstrap interval.
            Defaults to 0.05.
        n (int, optional): Number of bootstrap samples.
            Defaults to 10000.
    """
    # Reading data from a CSV file
    df = pd.read_csv(input_path)

    # Applying the dat_for_plot function to the data
    result = dat_for_plot(df, virus_type="T4", alpha=alpha, n=n)

    # Writing the results to a new CSV file
    result.to_csv(output_path, index=False)


def T7_efficacy_to_csv(
    input_path: str, output_path: str, alpha: float = 0.05, n: int = 10000
) -> None:
    """Function to calculate T7 phages from input file and write to output file results.

    Args:
        input_path (str): Path to csv with polony countings.
        output_path (str): Path to save results in csv file.
        alpha (float, optional): Confidence level for the bootstrap interval.
            Defaults to 0.05.
        n (int, optional): Number of bootstrap samples.
            Defaults to 10000.
    """
    # Reading data from a CSV file
    df = pd.read_csv(input_path)

    # Applying the dat_for_plot function to the data
    result = dat_for_plot(df, virus_type="T7", alpha=alpha, n=n)

    # Writing the results to a new CSV file
    result.to_csv(output_path, index=False)


def T7c_efficacy_to_csv(
    input_path: str, output_path: str, alpha: float = 0.05, n: int = 10000
) -> None:
    """Function to calculate T7c phages from input file and write to output file
        results.

    Args:
        input_path (str): Path to csv with polony countings.
        output_path (str): Path to save results in csv file.
        alpha (float, optional): Confidence level for the bootstrap interval.
            Defaults to 0.05.
        n (int, optional): Number of bootstrap samples.
            Defaults to 10000.
    """
    # Reading data from a CSV file
    df = pd.read_csv(input_path)

    # Applying the dat_for_plot function to the data
    result = dat_for_plot(df, virus_type="T7c", alpha=alpha, n=n)

    # Writing the results to a new CSV file
    result.to_csv(output_path, index=False)


def efficacy_to_csv(
    virus_type: str,
    input_path: str,
    output_path: str,
    alpha: float = 0.05,
    n: int = 10000,
) -> None:
    """Function to calculate phages according to virus type from input file and write to
        output file results.

    Args:
        virus_type (str): Type of virus. Can be T4, T7 or T7c.
        input_path (str): Path to csv with polony countings.
        output_path (str): Path to save results in csv file.
        alpha (float, optional): Confidence level for the bootstrap interval.
            Defaults to 0.05.
        n (int, optional): Number of bootstrap samples.
            Defaults to 10000.
    """
    if virus_type not in ["T4", "T7", "T7c"]:
        raise ValueError("Virus type should be T4, T7 or T7c")
    # Reading data from a CSV file
    df = pd.read_csv(input_path)

    # Applying the dat_for_plot function to the data
    result = dat_for_plot(df, virus_type=virus_type, alpha=alpha, n=n)

    # Writing the results to a new CSV file
    result.to_csv(output_path, index=False)
