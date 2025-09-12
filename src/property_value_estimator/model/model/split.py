import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

from property_value_estimator.core.config import TARGET_COLUMN

class ModelSplit(BaseModel):

    input_data: pd.DataFrame = Field(..., description="Input data for the model")
    random_state: int = Field(42, description="Random state for train/test split")
    test_split: float = Field(0.2, gt=0.0, lt=1.0, description="Proportion of the dataset to include in the test split")
    validation_split: float | None = Field(None, gt=0.0, lt=1.0, description="Proportion of the train dataset to include as validation data")
    feature_columns: list[str] | None = Field(None, description="List of feature column names, all columns by default")
    target_column: str = Field(default=TARGET_COLUMN, description="Name of the target column")

    class Config:
        arbitrary_types_allowed=True

    def process(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:

        target_column_data = self.input_data[self.target_column]
            
        feature_data = self.input_data[self._get_feature_columns()]


        X_train, X_test, y_train, y_test = train_test_split(
            feature_data.to_numpy(),
            target_column_data.to_numpy(),
            test_size=self.test_split,
            random_state=self.random_state
        )

        if self.validation_split:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validation_split, random_state=self.random_state)
        else:
            X_val, y_val = None, None

        train_data = self._recreate_dataset(X_train, y_train)
        test_data = self._recreate_dataset(X_test, y_test)
        val_data = self._recreate_dataset(X_val, y_val) if X_val is not None else None

        return train_data, test_data, val_data
    
    def _get_feature_columns(
            self
    ) -> list[str]:
        if self.feature_columns:
            assert all(col in self.input_data.columns for col in self.feature_columns), "Some feature columns are not in the input data"
            assert self.target_column not in self.feature_columns, "Target column should not be in feature columns"
            return self.feature_columns
        return self.input_data.columns.drop(self.target_column).tolist()

    def _recreate_dataset(
            self,
            features: np.ndarray,
            target: np.ndarray,
    ) -> pd.DataFrame:
        return pd.DataFrame(data=np.column_stack((features, target)), columns=self._get_feature_columns() + [self.target_column])
