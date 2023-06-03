from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

def getRealEstateData(spark):
    # Read the preprocessed CSV file into a DataFrame
    df = spark.read.csv("/data/real_estate_preprocessed.csv", header=True, inferSchema=True)
    return df

def trainModel(train_df):
    # Create a VectorAssembler to assemble the input columns into a vector column "features"
    assembler = VectorAssembler(inputCols=["area", "bedrooms", "bathrooms", "stories", "parking"], outputCol="features")

    # Transform the DataFrame by assembling the features
    train_df = assembler.transform(train_df)

    # Create a LinearRegression model
    lr = LinearRegression(featuresCol="features", labelCol="price")

    # Fit the model to the training DataFrame
    model = lr.fit(train_df)

    return model

def SaveModel(model):
    # Save the model with overwrite
    model.write().overwrite().save("/models/real_estate_model")
    print("Model saved at /models/real_estate_model")


def main():
    # Create a SparkSession
    spark = SparkSession.builder.appName("RealEstate").getOrCreate()

    # Load the real estate data
    df = getRealEstateData(spark)

    # Split the DataFrame into training and test sets
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Train the  model
    model = trainModel(train_df)

    # Save the  model
    SaveModel(model)

if __name__ == "__main__":
    main()
