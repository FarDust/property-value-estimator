"""
Simple API for property value estimation
"""

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from property_value_estimator.core.settings import settings
from property_value_estimator.infrastructure.db.entities.raw.zipcode_demographics import RawZipcodeDemographics
from property_value_estimator.infrastructure.models import ModelClient

# App
app = FastAPI(title="Property API")

# Database  
engine = create_engine(settings.database.uri)
SessionLocal = sessionmaker(bind=engine)

# Models
class PropertyInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int

class PredictionResponse(BaseModel):
    predicted_price: float
    zipcode: int

# Dependencies
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_model_client() -> ModelClient:
    return ModelClient()

# Helper functions
def get_demographics(db: Session, zipcode: int):
    demo = db.query(RawZipcodeDemographics).filter(RawZipcodeDemographics.zipcode == str(zipcode)).first()
    
    if not demo:
        raise ValueError(f"No demographic data found for zipcode {zipcode}")
    
    return {
        "medn_hshld_incm_amt": demo.medn_hshld_incm_amt,
        "per_urbn": demo.per_urbn,
        "per_sbrbn": demo.per_sbrbn,
        "per_bchlr": demo.per_bchlr,
        "per_prfsnl": demo.per_prfsnl,
    }

# Endpoints
@app.get("/")
async def root():
    return {"message": "Property API"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    property_input: PropertyInput,
    db: Session = Depends(get_db),
    model_client: ModelClient = Depends(get_model_client)
):
    # Get property data
    property_data = property_input.model_dump()
    
    # Get demographics
    demographics = get_demographics(db, property_input.zipcode)
    
    # Combine data
    combined_data = {**property_data, **demographics}
    
    # Call model
    predicted_price = await model_client.predict(combined_data)
    
    return PredictionResponse(
        predicted_price=predicted_price,
        zipcode=property_input.zipcode
    )

def main():
    """Main entry point for the API"""
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.debug
    )

if __name__ == "__main__":
    main()
