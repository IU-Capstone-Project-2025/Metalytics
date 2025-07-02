import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
from forecasting_models import ForecastModel
from typing import Dict, Any
from pyswarms.single import GlobalBestPSO


class Tuner(ABC):
    """
    Abstract class for model tuning.

    Attributes:
        model (ForecastModel): model to be tuned.
    """

    model: ForecastModel

    def __init__(self, model: ForecastModel):
        self.model = model

    @abstractmethod
    def tune(self) -> Dict[str, Any]:
        """
        Finds the suboptimal combination of hyperparameters.

        Returns:
            Dict[str, Any]: dictionary of hyperparameter names and values.
        """
        pass


class TuneXGBoost(Tuner):
    """PSO optimization for XGBoost hyperparameters"""
    df: pd.DataFrame

    n_particles: int = 5
    iters: int = 10

    bounds = (
        np.array([0.01, 3, 0.1, 0.1, 0.1, 0]),  # min values
        np.array([0.3, 10, 10, 1, 1, 5])        # max values
    )

    K = 20
    test_size: int = 24*10

    @property
    def metric(self):
        """Protected metric accessor"""
        return mean_squared_error

    def __init__(self, model: ForecastModel, df: pd.DataFrame):
        super().__init__(model)
        self.df = df

    def tune(self):

        def objective_function(params_list):
            scores = []
            for param_set in params_list:
                params = {
                    "learning_rate": param_set[0],
                    "max_depth": int(param_set[1]),
                    "min_child_weight": param_set[2],
                    "subsample": param_set[3],
                    "colsample_bytree": param_set[4],
                    "gamma": param_set[5]
                }

                scores.append(self.model.cross_validation(self.df, params, self.K, self.test_size, self.metric))

            return np.array(scores)

        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(self.bounds[0]),
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'early_stop': True, 'patience': 3},
            bounds=self.bounds
        )

        best_cost, best_params = optimizer.optimize(objective_function, iters=self.iters)

        optimized_params = {
            'learning_rate': best_params[0],
            'max_depth': int(best_params[1]),
            'min_child_weight': best_params[2],
            'subsample': best_params[3],
            'colsample_bytree': best_params[4],
            'gamma': best_params[5]
        }

        return optimized_params


class TuneLSTM(Tuner):
    """PSO optimization for LSTM hyperparameters"""
    df: pd.DataFrame

    n_particles: int = 5
    iters: int = 10

    bounds = (
        np.array([0.0001, 16, 16, 16, 0.1]),  # min values
        np.array([0.1, 256, 256, 256, 0.5])        # max values
    )

    K = 20
    test_size: int = 24*10

    @property
    def metric(self):
        """Protected metric accessor"""
        return mean_squared_error

    def __init__(self, model: ForecastModel, df: pd.DataFrame):
        super().__init__(model)
        self.df = df

    def tune(self):

        def objective_function(params_list):
            scores = []
            for param_set in params_list:
                params = {
                    "learning_rate": param_set[0],
                    "layer1": int(param_set[1]),
                    "layer2": int(param_set[2]),
                    "layer3": int(param_set[3]),
                    "dropout": param_set[4]
                }

                scores.append(self.model.cross_validation(self.df, params, self.K, self.test_size, self.metric))

            return np.array(scores)

        optimizer = GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=len(self.bounds[0]),
            options={'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'early_stop': True, 'patience': 3},
            bounds=self.bounds
        )

        best_cost, best_params = optimizer.optimize(objective_function, iters=self.iters)

        optimized_params = {
            "learning_rate": best_params[0],
            "layer1": int(best_params[1]),
            "layer2": int(best_params[2]),
            "layer3": int(best_params[3]),
            "dropout": best_params[4]
        }

        return optimized_params
