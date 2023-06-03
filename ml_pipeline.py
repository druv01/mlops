import subprocess

# Execute pre-processing.py
subprocess.run(["/opt/spark/bin/spark-submit", "pre-processing.py"])

# Execute train.py
subprocess.run(["/opt/spark/bin/spark-submit", "train.py"])

# Execute test.py
subprocess.run(["/opt/spark/bin/spark-submit", "test.py"])
