from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator

def getRealEstateData(spark):
    # Read the CSV file into a DataFrame
    df = spark.read.csv("/data/real_estate_test.csv", header=True, inferSchema=True)
    return df

def loadModel():
    
    # Load the saved model
    model = LinearRegressionModel.load("/models/real_estate_model")

    return model

def evaluateModel(model, df):

    # Create a VectorAssembler to assemble the input columns into a vector column "features"
    assembler = VectorAssembler(inputCols=["area", "bedrooms", "bathrooms", "stories", "parking"], outputCol="features")

    # Make predictions on the test DataFrame
    test_df = assembler.transform(df)
    predictions = model.transform(test_df)

    # Evaluate the model using a RegressionEvaluators
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    print("Root Mean Squared Error (RMSE) on test data = {:.2f}".format(rmse))

def main():
    
    # Create a SparkSession
    spark = SparkSession.builder.appName("RealEstate").getOrCreate()

    # Load the real estate data
    df = getRealEstateData(spark)

    # Evaluate the model
    evaluateModel(loadModel(), df)

    print("Testing the model is completed")


if __name__ == "__main__":
    main()
