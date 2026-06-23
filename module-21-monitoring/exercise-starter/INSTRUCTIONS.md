# Module 21 - API Monitoring with Prometheus and Grafana

## Overview

In this exercise you will instrument the Beans disease classifier API with Prometheus metrics. You will add custom metrics that track prediction confidence scores and per-class prediction counts, run the API with uvicorn, and verify the metrics endpoint directly in the workspace.

Running the full Prometheus + Grafana stack is optional and intended for local use only.


## 1. Complete `prometheus.yml`

Prometheus needs to know where to scrape metrics. Complete the configuration file so Prometheus scrapes the `/metrics` endpoint of the FastAPI app every 15 seconds.

Start Prometheus server using the configuration file
```bash
prometheus --config.file=prometheus.yml
```
---

## 2. Complete `main.py`

The API already serves predictions at `/predict`. You need to add Prometheus instrumentation.

**Add a Histogram** that tracks the distribution of prediction confidence scores per request. Use meaningful bucket boundaries between 0 and 1.

**Add a Counter** that tracks the number of predictions per output class, using the class name as a label.

**Instrument the FastAPI app** using `prometheus_fastapi_instrumentator` to automatically expose standard HTTP metrics (request count, latency) at `/metrics`. This is a different approach from the demo, which used a custom `@app.middleware("http")` to manually track these metrics. The instrumentator achieves the same result with less boilerplate.

**Inside `/predict`**: observe the confidence score in the histogram and increment the counter for the predicted class.

Read more: https://prometheus.github.io/client_python/

Launch the API
```
python main.py
```

Send requests to the API to simulate traffic
```bash
python send_requests.py
```
---

## 3. Launch Granafa and create the dashboard
```bash
export GF_SERVER_ROOT_URL=$(echo $VSCODE_PROXY_URI | sed "s/{{port}}/4000/")

grafana server --config=/etc/grafana/grafana.ini --homepath=/usr/share/grafana
```

Create a dashboard with panels for request rate, confidence distribution, and prediction counts per class
---
