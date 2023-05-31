from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
import matplotlib.pyplot as plt

def getRealEstateData(spark):
    df = spark.read.csv("/data/real_estate.csv", header=True, inferSchema=True) 
    return df

def linearRegression(df):
    assembler = VectorAssembler(inputCols=["sqft"], outputCol="features")
    df = assembler.transform(df)
    lr = LinearRegression(featuresCol="features", labelCol="price")
    model = lr.fit(df)

    # Save the model with overwrite
    model.write().overwrite().save("/models/real_estate_model")
    return model

def main():
    spark = SparkSession.builder.appName("RealEstate").getOrCreate()
    df = getRealEstateData(spark)
    model = linearRegression(df)
    print(f"Model saved successfully")

if __name__ == "__main__":
    main()
