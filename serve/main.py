from fastapi import FastAPI
from pydantic import BaseModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

app = FastAPI()

# Load the saved model
model = LinearRegressionModel.load("/models/real_estate_model")

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

class PredictionRequest(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    parking: int

class PredictionResponse(BaseModel):
    prediction: float

@app.get("/")
def home():
    return "Real estate predictions"

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_cols = ["area", "bedrooms", "bathrooms", "stories", "parking"]
    
    # Get the input data from the request
    area = request.area
    bedrooms = request.bedrooms
    bathrooms = request.bathrooms
    stories = request.stories
    parking = request.parking

    # Create a DataFrame from the input data
    input_data = spark.createDataFrame([(area, bedrooms, bathrooms, stories, parking)], input_cols)

    # Assemble features
    assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
    input_data = assembler.transform(input_data)

    # Make predictions
    predictions = model.transform(input_data)

    # Extract the prediction value
    prediction = predictions.select("prediction").first()[0]

    # Return the prediction as a response
    return PredictionResponse(prediction=prediction)

