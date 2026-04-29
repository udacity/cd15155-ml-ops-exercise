#TODO write the Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MLFLOW_TRACKING_URI="[http://host.docker.internal:5000](http://host.docker.internal:5000)"
ENV MLFLOW_MODEL_NAME="beans-disease-classifier"
ENV MLFLOW_MODEL_ALIAS="production"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
