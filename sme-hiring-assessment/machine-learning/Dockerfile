FROM python:3.10-slim
RUN pip install --upgrade pip
# Set the working directory in the container
WORKDIR /app
COPY . /app/
# Copy the dependencies file to the working directory
COPY model_output/ /app/model_output/
COPY requirements.txt /app/requirements.txt
COPY model.pkl /app/model.pkl
COPY model.py /app/model.py
# Copy the content of the local src directory to the working directory
COPY . .
# Install any dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN ls
# Specify the command to run on container start
EXPOSE 8080

# Define environment variable
ENV NAME MJassessment
# ENTRYPOINT ["python"]
ENTRYPOINT ["python", "main.py"]