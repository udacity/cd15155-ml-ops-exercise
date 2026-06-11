PREFECT_API_URL := http://127.0.0.1:4200/api

configure:
	prefect config set PREFECT_API_URL=$(PREFECT_API_URL)

start: configure
	mlflow server --port 5000 --host 0.0.0.0 --allowed-hosts "*" &
	prefect server start &
	sleep 5
	python -m prefect worker start --pool default &

stop:
	lsof -ti:5000 | xargs kill -9 || true
	lsof -ti:4200 | xargs kill -9 || true
	pkill -f "prefect worker" || true
