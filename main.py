from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI()

# Paths to the model, preprocessor, and encoders
MODEL_PATH = "./ML_Models/price/betal_price_prediction_model.pkl"
PREPROCESSOR_PATH = "./ML_Models/price/betel_preprocessing.pkl"
ENCODER_PATH = "./ML_Models/price/betel_label_encoders.pkl"

# Load the model, preprocessor, and encoders
try:
    loaded_model = joblib.load(MODEL_PATH)
    preprocessing_steps = joblib.load(PREPROCESSOR_PATH)  # Include any preprocessing logic, if needed
    label_encoders = joblib.load(ENCODER_PATH)  # Encoders for categorical variables
except Exception as e:
    raise RuntimeError(f"Error loading resources: {str(e)}")

# Input schema for price prediction
class PricePredictionInput(BaseModel):
    Date: str  # Format: YYYY-MM-DD
    Leaf_Type: str
    Leaf_Size: str
    Quality_Grade: str
    No_of_Leaves: int
    Location: str
    Season: str

# Prediction function
def predict_market_price(date, leaf_type, leaf_size, quality_grade, no_of_leaves, location, season):
    try:
        # Convert the date to a numeric month
        month = pd.to_datetime(date).month

        # Encode categorical features using the loaded encoders
        encoded_leaf_type = label_encoders['Leaf_Type'].transform([leaf_type])[0]
        encoded_leaf_size = label_encoders['Leaf_Size'].transform([leaf_size])[0]
        encoded_quality_grade = label_encoders['Quality_Grade'].transform([quality_grade])[0]
        encoded_location = label_encoders['Location'].transform([location])[0]
        encoded_season = label_encoders['Season'].transform([season])[0]

        # Prepare the feature vector
        features = [[month, encoded_leaf_type, encoded_leaf_size, encoded_quality_grade, no_of_leaves, encoded_location, encoded_season]]

        # Apply any preprocessing steps if required (e.g., scaling, transformations)
        if "scaler" in preprocessing_steps:
            features = preprocessing_steps["scaler"].transform(features)

        # Predict the price per leaf
        predicted_price = loaded_model.predict(features)[0]

        # Round the predicted price to the nearest 50 cents
        rounded_price = round(predicted_price * 2) / 2

        # Format the rounded price to two decimal places
        formatted_price = f"{rounded_price:.2f}"
        return formatted_price
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# Health Check Endpoint
@app.get("/health")
def health_check():
    """Health check endpoint to ensure the API is running."""
    return {"status": "API is up and running!"}

# Price Prediction Endpoint
@app.post("/predict-price")
def predict_price(input_data: PricePredictionInput):
    """
    Predict the market price per leaf based on input features.
    """
    try:
        # Perform prediction
        formatted_price = predict_market_price(
            date=input_data.Date,
            leaf_type=input_data.Leaf_Type,
            leaf_size=input_data.Leaf_Size,
            quality_grade=input_data.Quality_Grade,
            no_of_leaves=input_data.No_of_Leaves,
            location=input_data.Location,
            season=input_data.Season
        )
        return {"Predicted Market Price Per Leaf": formatted_price}
    except HTTPException as e:
        raise e  # Re-raise HTTP exceptions for clarity
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
