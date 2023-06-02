from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

def getRealEstateData(spark):
    # Read the CSV file into a DataFrame
    df = spark.read.csv("/data/real_estate.csv", header=True, inferSchema=True)
    return df

def linearRegression(train_df):
    # Create a VectorAssembler to assemble the input columns into a vector column "features"
    assembler = VectorAssembler(inputCols=["area", "bedrooms", "bathrooms", "stories", "parking"], outputCol="features")

    # Transform the DataFrame by assembling the features
    train_df = assembler.transform(train_df)

    # Create a LinearRegression model
    lr = LinearRegression(featuresCol="features", labelCol="price")

    # Fit the model to the training DataFrame
    model = lr.fit(train_df)

    # Save the model with overwrite
    model.write().overwrite().save("/models/real_estate_model")

    return model, assembler

def evaluateModel(model, assembler, test_df):
    # Make predictions on the test DataFrame
    test_df = assembler.transform(test_df)
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

    # Split the DataFrame into training and test sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Train the linear regression model
    model, assembler = linearRegression(train_df)

    # Evaluate the model
    evaluateModel(model, assembler, test_df)

if __name__ == "__main__":
    main()
