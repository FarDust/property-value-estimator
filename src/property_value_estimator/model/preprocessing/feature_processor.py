import pandas as pd


from pydantic import BaseModel, Field

class FeatureProcessor(BaseModel):
    """
    Feature selection and preparation for training/evaluation, following ModelTrainer pattern.
    """
    data: pd.DataFrame = Field(..., description="Input DataFrame")
    target_column: str = Field(..., description="Name of the target column")
    feature_columns: list[str] | None = Field(None, description="List of feature column names, all columns by default")

    class Config:
        arbitrary_types_allowed = True

    def get_feature_columns(self) -> list[str]:
        """
        Get feature column names, following the same pattern as ModelTrainer.
        """
        if self.feature_columns:
            assert all(col in self.data.columns for col in self.feature_columns), "Some feature columns are not in the data"
            assert self.target_column not in self.feature_columns, "Target column should not be in feature columns"
            return self.feature_columns
        return self.data.columns.drop(self.target_column).tolist()


    def get_feature_target_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame containing both the selected feature columns and the target column.
        """
        cols = self.get_feature_columns() + [self.target_column]
        return self.data[cols]
