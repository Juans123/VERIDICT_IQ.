.PHONY: install run test clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	python run_demo.py

test:
	pytest -q

clean:
	python -c "import shutil; shutil.rmtree('artifacts', ignore_errors=True)"
