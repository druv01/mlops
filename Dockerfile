FROM apache/spark-py

# Set the user to "root"
USER root

# Install any required dependencies
RUN pip install numpy matplotlib pandas

# Set the working directory to /root/project
WORKDIR /root/project

# Copy the python files to the container
COPY pre-processing.py train.py test.py .

# Create a volume mount to store the model file
VOLUME /models

# Create a volume mount to store the data file
VOLUME /data

# Specify the command to run spark-submit
CMD ["/opt/spark/bin/spark-submit", "pre-processing.py", "train.py", "test.py"]
