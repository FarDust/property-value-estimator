"""
Model client for MLflow REST API calls
"""

import httpx
from typing import Dict, Any
from property_value_estimator.core.settings import settings


class ModelClient:
    """Client for MLflow model service REST API calls"""
    
    def __init__(self):
        self.base_url = settings.model_service.url
        self.timeout = settings.model_service.timeout
    
    async def predict(self, data: Dict[str, Any]) -> float:
        """Call MLflow model service for prediction"""
        payload = {
            "dataframe_split": {
                "columns": list(data.keys()),
                "data": [list(data.values())]
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/invocations",
                json=payload
            )
            
            result = response.json()
            if isinstance(result, list):
                return float(result[0])
            if "predictions" not in result:
                raise KeyError("'predictions' key missing in MLflow response: {}".format(result))
            return float(result["predictions"][0])
