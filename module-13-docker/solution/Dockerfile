#TODO write the Dockerfile

#TODO Define the base image (Python 3.12 image)
FROM python:3.12-slim

#TODO Set the working directory in the container
WORKDIR /app

#TODO Copy the requirements.txt file to the container and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#TODO Copy the rest of the application code to the container
COPY . .

#TODO Set environment variables for MLflow tracking URI, model name and model alias
ENV MLFLOW_TRACKING_URI="http://host.docker.internal:5000"
ENV MLFLOW_MODEL_NAME="beans-disease-classifier"
ENV MLFLOW_MODEL_ALIAS="production"

#TODO Expose the port that the FastAPI app will run on
EXPOSE 8000

#TODO Define the command to run the FastAPI app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/proxy/8000"]
