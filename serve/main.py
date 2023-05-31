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
    sqft: float

class PredictionResponse(BaseModel):
    prediction: float

@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Get the input data from the request
    sqft = request.sqft

    # Create a DataFrame from the input data
    input_data = spark.createDataFrame([(sqft,)], ["sqft"])

    # Assemble features
    assembler = VectorAssembler(inputCols=["sqft"], outputCol="features")
    input_data = assembler.transform(input_data)

    # Make predictions
    predictions = model.transform(input_data)

    # Extract the prediction value
    prediction = predictions.select("prediction").first()[0]

    # Return the prediction as a response
    return PredictionResponse(prediction=prediction)
