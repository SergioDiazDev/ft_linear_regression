#Create venv
venv:
	python3 -m venv src/venv
	. src/venv/bin/activate
	@echo "Virtual environment created"
	@echo "To activate the virtual environment, run: /"source src/venv/bin/activate/""
	@echo "To install the dependencies, run: /"make install/""

install:
	pip install -r src/requirements.txt
	@echo "Dependencies installed"

