########################################################################################################################
# Model Building
########################################################################################################################
# Author
#   David Cavanaugh
#
# Created Date
#   2021-12-06
#
# Description
#   Building the models for the given data
#
########################################################################################################################
# BEGIN WORKING DOCUMENT
########################################################################################################################
# Import statements
from typing import Optional, Union
from generics import Model
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVR, LinearSVR

from catboost import CatBoostRegressor, Pool


########################################################################################################################
# Functions
def modeller(y: pd.Series, x: Union[pd.DataFrame, Pool], spec: dict, cat_feats: Optional[list] = None,
             final_allocation: Optional[pd.Series] = None) -> Model:
    """Modeller

    Parameters
    ----------
    y: pd.Series
        Target column
    x: pd.DataFrame
        Data to use as inputs
    spec: dict
        Model specification
    cat_feats: list, default None
        Categorical features to utilize in the model - only used for CatBoost models
    final_allocation: pd.Series, default None
        The column for final allocation - if it is not used in the data and the target variable is Used_Amount

    Returns
    -------
    model: Model
        Fitted model
    """
    framework = spec.get('framework')
    params = spec.get('params')
    assert framework is not None and params is not None, \
        "`spec` must contain both a framework (string) and the model params (dict)"

    if framework == "GradientBoostingRegressor":
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = GradientBoostingRegressor(**params)
        model.fit(X=x, y=y)
    elif framework == 'KNeighborsRegressor':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = KNeighborsRegressor(**params)
        model.fit(X=x, y=y)
    elif framework == 'MultinomialNB':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = MultinomialNB(**params)
        model.fit(X=x, y=y)
    elif framework == 'GaussianNB':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = GaussianNB(**params)
        model.fit(X=x, y=y)
    elif framework == 'SVR':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = SVR(**params)
        model.fit(X=x, y=y)
    elif framework == 'LinearSVR':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = LinearSVR(**params)
        model.fit(X=x, y=y)
    elif framework == 'RandomForestRegressor':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = RandomForestRegressor(**params)
        model.fit(X=x, y=y)
    elif framework == 'AdaBoostRegressor':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = AdaBoostRegressor(**params)
        model.fit(X=x, y=y)
    elif framework == 'ExtraTreesRegressor':
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = ExtraTreesRegressor(**params)
        model.fit(X=x, y=y)
    elif framework == "LinearRegression":
        assert y is not None, f"When using {framework}, `y` must be specified"
        model = LinearRegression(**params)
        model.fit(X=x, y=y)
    elif framework == 'CatBoostRegressor':
        pool = Pool(data=x, label=y, cat_features=cat_feats)
        model = CatBoostRegressor(**params)
        model.fit(X=pool)
    else:
        raise ValueError(f"Unrecognized model framework {framework}")

    return Model(model, spec)


########################################################################################################################
# END WORKING DOCUMENT
########################################################################################################################
