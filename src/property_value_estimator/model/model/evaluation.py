from typing import Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.graph_objects import Figure
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance

from property_value_estimator.core.config import TARGET_COLUMN


class ChartConfig(BaseModel):
    """Configuration for chart generation"""
    width: int = Field(default=800, description="Chart width in pixels")
    height: int = Field(default=600, description="Chart height in pixels")
    template: str = Field(default="plotly_white", description="Plotly template theme")


class ModelEvaluator(BaseModel):
    """Model evaluation class following the project pattern"""
    
    model: Any = Field(..., description="Trained model to evaluate")
    testing_data: pd.DataFrame = Field(..., description="Testing dataset")
    target_column: str = Field(default=TARGET_COLUMN, description="Name of the target column")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds for learning curves")
    chart_config: ChartConfig = Field(default_factory=ChartConfig, description="Chart configuration")

    class Config:
        arbitrary_types_allowed=True

    def evaluate(self) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Figure]]:
        """
        Evaluate the model and return metrics, feature importance, and figures
        
        Returns:
            tuple: (metrics_df, feature_importance_df, figures_dict)
        """
        # Prepare data
        X_test, y_test = self._prepare_testing_data()
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics_df = self._calculate_regression_metrics(y_test, y_pred)
        
        # Calculate feature importance
        feature_importance_df = self._calculate_feature_importance(X_test, y_test)
        
        # Generate figures
        figures_dict = self._generate_figures(X_test, y_test, y_pred, feature_importance_df)
        
        return metrics_df, feature_importance_df, figures_dict

    def _prepare_testing_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare testing data for evaluation"""
        y_test = self.testing_data[self.target_column].values
        X_test = self.testing_data.drop(columns=[self.target_column]).values
        return X_test, y_test

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
        """Calculate regression metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics_df = pd.DataFrame({
            'metric': ['mae', 'rmse', 'r2', 'mape'],
            'value': [mae, rmse, r2, mape],
            'description': [
                'Mean Absolute Error',
                'Root Mean Square Error', 
                'R-squared',
                'Mean Absolute Percentage Error'
            ]
        })
        
        return metrics_df

    def _calculate_feature_importance(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Calculate feature importance using permutation importance"""
        feature_names = self.testing_data.drop(columns=[self.target_column]).columns.tolist()
        
        # Use permutation importance as it works with any model
        perm_importance = permutation_importance(
            self.model, X_test, y_test, 
            random_state=self.random_state,
            n_repeats=10
        )
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Add rank
        feature_importance_df['rank'] = range(1, len(feature_importance_df) + 1)
        
        return feature_importance_df

    def _generate_figures(self, X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, 
                         feature_importance_df: pd.DataFrame) -> dict[str, Figure]:
        """Generate all evaluation figures"""
        figures = {}
        
        figures['prediction_vs_actual'] = self._create_prediction_vs_actual_plot(y_test, y_pred)
        figures['residuals'] = self._create_residuals_plot(y_test, y_pred)
        figures['feature_importance'] = self._create_feature_importance_plot(feature_importance_df)
        figures['learning_curves'] = self._create_learning_curves(X_test, y_test)
        
        return figures

    def _create_prediction_vs_actual_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
        """Create prediction vs actual values scatter plot"""
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker=dict(opacity=0.6, size=8),
            name='Predictions',
            hovertemplate='<b>Actual:</b> %{x:,.0f}<br><b>Predicted:</b> %{y:,.0f}<extra></extra>'
        ))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Prediction',
            hovertemplate='Perfect Prediction Line<extra></extra>'
        ))
        
        fig.update_layout(
            title='Predictions vs Actual Values',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            template=self.chart_config.template,
            width=self.chart_config.width,
            height=self.chart_config.height,
            showlegend=True
        )
            
        return fig

    def _create_residuals_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> Figure:
        """Create residuals plot"""
        residuals = y_true - y_pred
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(opacity=0.6, size=8),
            name='Residuals',
            hovertemplate='<b>Predicted:</b> %{x:,.0f}<br><b>Residual:</b> %{y:,.0f}<extra></extra>'
        ))
        
        # Zero line
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            annotation_text="Zero Line"
        )
        
        fig.update_layout(
            title='Residuals Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals (Actual - Predicted)',
            template=self.chart_config.template,
            width=self.chart_config.width,
            height=self.chart_config.height
        )
            
        return fig

    def _create_feature_importance_plot(self, feature_importance_df: pd.DataFrame) -> Figure:
        """Create feature importance bar plot"""
        
        # Plot top 10 features
        top_features = feature_importance_df.head(10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_features['feature'],
            x=top_features['importance'],
            orientation='h',
            error_x=dict(
                type='data',
                array=top_features['std'] if 'std' in top_features.columns else None,
                visible=True
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
            marker_color='lightblue',
            marker_line_color='navy',
            marker_line_width=1
        ))
        
        fig.update_layout(
            title='Feature Importance (Top 10)',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template=self.chart_config.template,
            width=self.chart_config.width,
            height=self.chart_config.height,
            yaxis={'categoryorder': 'total ascending'}
        )
            
        return fig

    def _create_learning_curves(self, X_test: np.ndarray, y_test: np.ndarray) -> Figure:
        """Create learning curves plot"""
        
        # Generate learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_test, y_test, 
            cv=self.cv_folds,
            random_state=self.random_state,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_absolute_error'
        )
        
        # Convert to positive values (MAE)
        train_scores = -train_scores
        val_scores = -val_scores
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training Score',
            line=dict(color='blue'),
            hovertemplate='<b>Training Size:</b> %{x}<br><b>MAE:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Training confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([train_mean - train_std, (train_mean + train_std)[::-1]]),
            fill='tonext',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Training ±1 std',
            showlegend=False
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Validation Score',
            line=dict(color='red'),
            hovertemplate='<b>Training Size:</b> %{x}<br><b>MAE:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Validation confidence interval
        fig.add_trace(go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([val_mean - val_std, (val_mean + val_std)[::-1]]),
            fill='tonext',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Validation ±1 std',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Learning Curves',
            xaxis_title='Training Set Size',
            yaxis_title='Mean Absolute Error',
            template=self.chart_config.template,
            width=self.chart_config.width,
            height=self.chart_config.height,
            showlegend=True
        )
            
        return fig