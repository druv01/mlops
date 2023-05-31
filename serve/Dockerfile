# Base image with Spark and Python
FROM apache/spark-py

# Set the user to "root"
USER root

# Install additional dependencies
RUN pip install fastapi uvicorn numpy pandas
RUN pip install pyspark

# Set working directory
#WORKDIR /app

# Copy the FastAPI application code
COPY main.py .
COPY submit.py .

# Create a volume mount to store the model file
VOLUME /models

# Expose the application port
EXPOSE 8000

# Command to start the FastAPI application
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# Set the entrypoint to spark-submit
ENTRYPOINT ["/opt/spark/bin/spark-submit"]

# Specify the command and arguments
CMD ["--master", "local[*]", "submit.py"]
