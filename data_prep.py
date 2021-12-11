########################################################################################################################
# Data Prep
########################################################################################################################
# Author
#   David Cavanaugh
#
# Created Date
#   2021-12-06
#
# Description
#   Functions which prepare the data for future modelling
#
########################################################################################################################
# BEGIN WORKING DOCUMENT
########################################################################################################################
# Import Statements
from typing import Tuple, Optional
import pandas as pd


########################################################################################################################
# Functions
def read_train_data() -> pd.DataFrame():
    """Read Data

    Returns
    -------
    df: pd.DataFrame
        Raw train data
    """
    df = pd.read_csv("./Data/train.csv", parse_dates=["StartDate_award", "EndDate_award", "StartDate_usage",
                                                      "EndDate_usage"]).set_index('Grant_Number')
    return df


def read_test_data() -> pd.DataFrame():
    """Read Test Data

    Returns
    -------
    df: pd.DataFrame
        Raw test data
    """
    df = pd.read_csv("./Data/test.csv", parse_dates=["StartDate_award", "EndDate_award", "StartDate_usage",
                                                     "EndDate_usage"]).set_index('Grant_Number')
    return df


def create_Xy(data: pd.DataFrame, y_col: str = "UsageRate",
              x_cols: Optional[list] = None) -> Tuple[pd.Series, pd.DataFrame]:
    """Create X and y Data Sets

    Note: Remove the equivalent variables for the y (target) variable

    Parameters
    ----------
    data: pd.DataFrame
        Data to create Xy from
    y_col: str, default 'UsageRate'
        column to use as the target (y) variable
    x_cols: list, default None
        Columns to use in the X dataframe

    Returns
    -------
    y: pd.Series
        y variable
    x: pd.DataFrame
        X variables
    """
    assert y_col in ["Used_Amount", "UsageRate"], "`y_col` not one of the accepted target variables"

    y = data[y_col]

    x = data.drop([y_col], axis=1)
    if x_cols:
        x = x[x_cols]

    if y_col == "UsageRate" and "Used_Amount" in x.columns:
        x.drop(["Used_Amount"], axis=1, inplace=True)
    elif y_col == "UsageRate" and "UsageRate" in x.columns:
        x.drop(["UsageRate"], axis=1, inplace=True)

    return y, x






