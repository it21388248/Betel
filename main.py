from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Initialize FastAPI app
app = FastAPI()

# Load the saved model
model_path = "./ML_Models/price/betal_price_prediction_model.pkl"  # Update the path as per your setup
loaded_model = joblib.load(model_path)

# Prepare label encoders for categorical columns
categorical_columns = ['Leaf_Type', 'Leaf_Size', 'Quality_Grade', 'Location', 'Season']
label_encoders = {col: LabelEncoder() for col in categorical_columns}

# Manually set the encoders with the same mappings as during training
leaf_type_classes = ['Peedichcha', 'Korikan', 'Keti', 'Raan Keti']
leaf_size_classes = ['Small', 'Medium', 'Large']
quality_grade_classes = ['Ash', 'Dark']
location_classes = ['Kuliyapitiya', 'Naiwala', 'Apaladeniya']
season_classes = ['Dry', 'Rainy']

label_encoders['Leaf_Type'].fit(leaf_type_classes)
label_encoders['Leaf_Size'].fit(leaf_size_classes)
label_encoders['Quality_Grade'].fit(quality_grade_classes)
label_encoders['Location'].fit(location_classes)
label_encoders['Season'].fit(season_classes)

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
def predict_market_price_rounded(date, leaf_type, leaf_size, quality_grade, no_of_leaves, location, season):
    try:
        # Convert the date to a numeric month
        month = pd.to_datetime(date).month

        # Encode categorical features
        encoded_leaf_type = label_encoders['Leaf_Type'].transform([leaf_type])[0]
        encoded_leaf_size = label_encoders['Leaf_Size'].transform([leaf_size])[0]
        encoded_quality_grade = label_encoders['Quality_Grade'].transform([quality_grade])[0]
        encoded_location = label_encoders['Location'].transform([location])[0]
        encoded_season = label_encoders['Season'].transform([season])[0]

        # Prepare the feature vector
        features = [[month, encoded_leaf_type, encoded_leaf_size, encoded_quality_grade, no_of_leaves, encoded_location, encoded_season]]

        # Predict the price per leaf
        predicted_price = loaded_model.predict(features)[0]

        # Round the predicted price to the nearest 50 cents
        rounded_price = round(predicted_price * 2) / 2

        # Format the rounded price to two decimal places
        formatted_price = f"{rounded_price:.2f}"
        return formatted_price
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# Price Prediction Endpoint
@app.post("/predict-price")
def predict_price(input_data: PricePredictionInput):
    try:
        formatted_price = predict_market_price_rounded(
            date=input_data.Date,
            leaf_type=input_data.Leaf_Type,
            leaf_size=input_data.Leaf_Size,
            quality_grade=input_data.Quality_Grade,
            no_of_leaves=input_data.No_of_Leaves,
            location=input_data.Location,
            season=input_data.Season
        )
        return {"Predicted Market Price Per Leaf": formatted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "API is up and running!"}
