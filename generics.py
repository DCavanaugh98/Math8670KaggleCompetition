########################################################################################################################
# Generics
########################################################################################################################
# Author
#   David Cavanaugh
#
# Created Date
#   2021-12-06
#
# Description
#   Home for generic wrappers for multiple different model frameworks
#
########################################################################################################################
# BEGIN WORKING DOCUMENT
########################################################################################################################
# Import Statements
from typing import Optional
import numpy as np
from sklearn.metrics import mean_squared_error

########################################################################################################################
import pandas as pd


class Model:
    """
    Generic Model Class
    """
    def __init__(self, model, spec):
        self.model = model
        self.framework = spec.get('framework')
        self.y_var = spec.get('y_var')

        self.train, self.test = None, None
        self.train_y, self.test_y = None, None
        self.train_pred, self.test_pred = None, None
        self.train_final_alloc, self.test_final_alloc = None, None

        self.train_mse, self.test_mse = None, None

        self.metric_df = None

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, train_final_alloc: Optional[pd.Series] = None,
                test_final_alloc: Optional[pd.Series] = None):
        self.train = train
        self.test = test
        if self.framework in ['GradientBoostingRegressor', "CatBoostRegressor", "KNeighborsRegressor",
                              "GaussianNB", "MultinomialNB", "SVR", "LinearSVR", "RandomForestRegressor",
                              "ExtraTreesRegressor", "AdaBoostRegressor", "LinearRegression"]:
            self.train_pred = self.model.predict(train)
            self.test_pred = self.model.predict(test)
        else:
            raise ValueError(f"Unrecognized framework {self.framework}")

        if self.y_var == 'Used_Amount':
            self.train_final_alloc = train_final_alloc
            self.test_final_alloc = test_final_alloc
            self.train_pred = np.clip(self.train_pred / train_final_alloc, a_min=None, a_max=6)
            self.test_pred = np.clip(self.test_pred / test_final_alloc, a_min=None, a_max=6)

    def evaluate(self, train_y, test_y):
        if self.y_var == 'Used_Amount':
            self.train_y = np.clip(train_y / self.train_final_alloc, a_min=None, a_max=6)
            self.test_y = np.clip(test_y / self.test_final_alloc, a_min=None, a_max=6)
        else:
            self.train_y = train_y
            self.test_y = test_y
        self.train_mse = mean_squared_error(self.train_y, self.train_pred)
        self.test_mse = mean_squared_error(self.test_y, self.test_pred)

        self.metric_df = pd.DataFrame({'Train': [self.train_mse], 'Test': [self.test_mse]},
                                      index=['Mean Squared Error'])

        return self.metric_df

########################################################################################################################
# END WORKING DOCUMENT
########################################################################################################################
