from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler

def getRealEstateData(spark):
    # Read the CSV file into a DataFrame
    df = spark.read.csv("/data/real_estate.csv", header=True, inferSchema=True)
    return df

def preprocessData(df):
    # Drop any rows with missing values
    df = df.dropna()

    # Remove duplicates
    df = df.dropDuplicates()

    # Perform any other necessary data cleaning steps

    return df

def saveAsCSV(df, path):
    # Save the DataFrame as a CSV file
    df.write.csv(path, header=True, mode="overwrite")
    print(f"Data saved at {path}")

def main():
    # Create a SparkSession
    spark = SparkSession.builder.getOrCreate()

    # Load the real estate data
    real_estate_df = getRealEstateData(spark)

    # Perform data preprocessing
    preprocessed_df = preprocessData(real_estate_df)

    # Save the preprocessed data as a CSV file
    saveAsCSV(preprocessed_df, "/data/real_estate_preprocessed.csv")

if __name__ == "__main__":
    main()
