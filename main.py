from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib


app=FastAPI()

COLUMNS=["latitude","longitude","price","minimum_nights","number_of_reviews","reviews_per_month","calculated_host_listings_count","availability_365",""
        "availability_365","neighbourhood_group","neighbourhood"]

model=joblib.load("model_pipeline.pkl")

# Pydantic model = the input is validation

class Features(BaseModel):
    latitude: float = Field(...,ge=-90,le=90,description="Latitude Coordinate")
    longitude: float = Field(...,ge=-180,le=180,description="Longitude Coordinate")
    price: float = Field(...,gt=0,description="Price Per Night Must be Positive")
    minimum_nights:int = Field(...,ge=1,le=365,description="Minimum Night required for Booking")
    number_of_reviews: int = Field(...,ge=0,description="Total Number of Reviews")
    reviews_per_month: float = Field(...,ge=0,description="Average reviews per month")
    calculated_host_listings_count: int = Field(...,ge=0,description="Number of listing by the host")
    availability_365: int = Field(...,ge=0,le=365,description="Days Available out of 365")
    neighbourhood_group:str = Field(...,min_length=1,description="Borough or neighbourhood gruop")
    neighbourhood: str = Field(...,min_length=1,description="Specific neighbourhood name")

@app.get('/')

def greet():
    return "Hello Guys"

@app.post('/predict')
def predict(features):
    row=pd.DataFrame(features.dict(),columns=COLUMNS)
    prediction=model.predict(row)
    probability=model.predict_proba(row)

    return {
        "Prediction_room_type":prediction.tolist(),
        "probability":probability.tolist()}
    