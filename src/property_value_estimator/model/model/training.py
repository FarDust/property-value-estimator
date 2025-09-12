from typing import Any
from pydantic import BaseModel, PrivateAttr, Field
from sklearn import neighbors
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn import preprocessing

from property_value_estimator.core.config import TARGET_COLUMN


class ModelTrainer(BaseModel):
    model_name: str = Field(..., description="Name for the trained model")
    input_data: pd.DataFrame = Field(..., description="Input data for training")
    target_column: str = Field(default=TARGET_COLUMN, description="Name of the target column")
    feature_columns: list[str] | None = Field(None, description="List of feature column names, all columns by default")

    _scaler: Any | None = PrivateAttr(None)
    _regressor: Any | None = PrivateAttr(None)

    class Config:
        arbitrary_types_allowed=True

    @property
    def scaler(self):
        if not self._scaler:
            self._scaler = preprocessing.RobustScaler()
        return self._scaler

    @property
    def regressor(self):
        if not self._regressor:
            self._regressor = neighbors.KNeighborsRegressor()
        return self._regressor

    def get_feature_columns(self) -> list[str]:
        """Get feature column names, following the same pattern as ModelSplit"""
        if self.feature_columns:
            assert all(col in self.input_data.columns for col in self.feature_columns), "Some feature columns are not in the input data"
            assert self.target_column not in self.feature_columns, "Target column should not be in feature columns"
            return self.feature_columns
        return self.input_data.columns.drop(self.target_column).tolist()

    def _prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data as numpy arrays, following the same pattern as ModelSplit"""
        target_data = self.input_data[self.target_column]
        feature_data = self.input_data[self.get_feature_columns()]
        
        return feature_data.to_numpy(), target_data.to_numpy()

    def train(self):
        """Train the model using the prepared data"""
        X_train, y_train = self._prepare_training_data()
        
        model = Pipeline([
            ('scaler', self.scaler),
            ('regressor', self.regressor)
        ]).fit(X_train, y_train)

        return model
