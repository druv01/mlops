from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("RealEstatePrediction").getOrCreate()

# Set any additional Spark configurations if needed
# spark.conf.set("spark.some.config.option", "value")

# Submit main.py as a Spark application
spark.sparkContext.addPyFile("main.py")  # Add main.py to the Spark context
spark.sparkContext.setLogLevel("INFO")  # Set the log level if needed

# Import main module and run the FastAPI application
from main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
