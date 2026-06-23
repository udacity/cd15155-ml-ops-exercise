PREFECT_API_URL := http://127.0.0.1:4200/api
export PREFECT_UI_API_URL := $(shell echo $$VSCODE_PROXY_URI | sed 's/{{port}}/4200/')api

configure:
	prefect config set PREFECT_API_URL=$(PREFECT_API_URL)

start: configure
	mlflow server --port 5000 --host 0.0.0.0 --allowed-hosts "*" &
	prefect server start --host 0.0.0.0 &
	sleep 5
	python -m prefect worker start --pool default &

stop:
	pkill -9 -f "[m]lflow" || true
	pkill -9 -f "[p]refect server" || true
	pkill -9 -f "[p]refect worker" || true
