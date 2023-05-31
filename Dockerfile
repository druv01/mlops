FROM apache/spark-py

# Set the user to "root"
USER root

# Install any required dependencies
RUN pip install numpy matplotlib pandas

# Set the working directory to /root/project
WORKDIR /root/project

# Copy the real_estate.py file to the container
COPY real_estate.py .

# Create a volume mount to store the model file
VOLUME /models

# Create a volume mount to store the data file
VOLUME /data

# Specify the command to run spark-submit
CMD ["/opt/spark/bin/spark-submit", "real_estate.py"]
