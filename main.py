from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

import firebase_admin
from firebase_admin import credentials, firestore, auth

# Initialize the FastAPI app
app = FastAPI()

# Path to your Firebase Admin SDK JSON file
cred = credentials.Certificate("C:/Users/Kavindi/Downloads/betelapp-1d34b-firebase-adminsdk-z5vqk-1605baec1b.json")

# Initialize the Firebase Admin SDK
firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Server is up and running!"}

# Sample Firestore Route
@app.get("/")
def index():
    # Initialize a Firestore document reference
    doc_ref = db.collection('test').document('sample')

    # Add a sample document to Firestore
    doc_ref.set({
        'name': 'John Doe',
        'role': 'Seller'
    })

    return {"message": "Document added!"}

# Load models, preprocessors, and encoders
demand_model = joblib.load('./ML_Models/demand/betel_demand_forecasting_model.pkl')
demand_preprocessor = joblib.load('./ML_Models/demand/betel_demand_preprocessor.pkl')
demand_label_encoder = joblib.load('./ML_Models/demand/betel_demand_label_encoder.pkl')

price_model = joblib.load('./ML_Models/price/price_prediction_model.pkl')
price_preprocessor = joblib.load('./ML_Models/price/price_preprocessor.pkl')

# Define the input schema for demand prediction
class InputData(BaseModel):
    Leaf_Type: str
    Leaf_Size: str
    Leaf_Condition: str
    Quality_Grade: str
    Season: str
    Disease_Impact: str
    District: str
    City: str
    Harvest_Quantity: int
    Transportation_Cost: int
    Year: int
    Month: int
    Day: int

# Define the input schema for price prediction
class PricePredictionInput(BaseModel):
    Date: str  # In DD/MM/YYYY format
    Leaf_Type: str
    Leaf_Size: str
    Leaf_Condition: str
    Quality_Grade: str
    Season: str
    Disease_Impact: str
    Market_Demand: str
    Harvest_Quantity: int
    Transportation_Cost: int
    District: str
    City: str

@app.post("/predict-market-demand")
def predict_market_demand(input_data: InputData):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Apply preprocessing
        encoded_input = demand_preprocessor.transform(input_df)

        # Predict market demand
        predicted_demand_encoded = demand_model.predict(encoded_input)

        # Decode the predicted demand
        predicted_demand = demand_label_encoder.inverse_transform(predicted_demand_encoded)

        return {"Predicted Market Demand": predicted_demand[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-price")
def predict_price(input_data: PricePredictionInput):
    try:
        # Convert input data to dictionary
        input_dict = input_data.dict()

        # Extract Year, Month, and Day from the Date string
        input_dict['Year'] = int(input_dict['Date'].split('/')[2])
        input_dict['Month'] = int(input_dict['Date'].split('/')[1])
        input_dict['Day'] = int(input_dict['Date'].split('/')[0])
        input_dict.pop('Date')

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Apply preprocessor
        encoded_input = price_preprocessor.transform(input_df)

        # Predict price
        predicted_price = price_model.predict(encoded_input)

        return {"Predicted Betel Leaf Price (per leaf)": f"{predicted_price[0]:.2f}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Chatbot endpoint
class ChatRequest(BaseModel):
    message: str  # User's input message

@app.post("/chatbot")
def chatbot_response(input_data: ChatRequest):
    try:
        # Get the user's message
        user_message = input_data.message.lower()

        # Determine intent (price or market demand)
        if "price" in user_message:
            # Example input for price prediction
            input_dict = {
                "Date": "02/04/2025",
                "Leaf_Type": "Maneru",
                "Leaf_Size": "Medium",
                "Leaf_Condition": "Fresh",
                "Quality_Grade": "High",
                "Season": "Dry",
                "Disease_Impact": "No",
                "Market_Demand": "High",
                "Harvest_Quantity": 3000,
                "Transportation_Cost": 1500,
                "District": "Gampaha",
                "City": "Mirigama"
            }

            input_dict['Year'] = int(input_dict['Date'].split('/')[2])
            input_dict['Month'] = int(input_dict['Date'].split('/')[1])
            input_dict['Day'] = int(input_dict['Date'].split('/')[0])
            input_dict.pop('Date')

            input_df = pd.DataFrame([input_dict])

            # Preprocess input
            encoded_input = price_preprocessor.transform(input_df)

            # Predict price
            predicted_price = price_model.predict(encoded_input)[0]

            return {"response": f"The predicted price for betel leaves is {predicted_price:.2f} per leaf."}

        elif "demand" in user_message:
            # Example input for market demand prediction
            input_dict = {
                "Leaf_Type": "Maneru",
                "Leaf_Size": "Medium",
                "Leaf_Condition": "Fresh",
                "Quality_Grade": "High",
                "Season": "Dry",
                "Disease_Impact": "No",
                "District": "Gampaha",
                "City": "Mirigama",
                "Harvest_Quantity": 3000,
                "Transportation_Cost": 1500,
                "Year": 2025,
                "Month": 4,
                "Day": 2
            }

            input_df = pd.DataFrame([input_dict])

            # Preprocess input
            encoded_input = demand_preprocessor.transform(input_df)

            # Predict demand
            predicted_demand_encoded = demand_model.predict(encoded_input)
            predicted_demand = demand_label_encoder.inverse_transform(predicted_demand_encoded)[0]

            return {"response": f"The predicted market demand is {predicted_demand}."}

        else:
            # Fallback response for unknown intents
            return {"response": "I can help you with price predictions or market demand. Please specify your query."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

