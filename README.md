# ft_linear_regression

#### This project implements linear regression using gradient descent to predict car prices based on the number of kilometers driven. The model iteratively optimizes its parameters to minimize the cost function and improve price predictions.

### venv:

	make venv
	source src/venv/bin/activate
	make install

### Run:
	(venv)$ python3 src/train.py -c -g -l 5e-1 -e 15 -m model.pkl

"model.pkl" is save in the folder "src/models"

-h: For more Info

	(venv)$ python3 src/predict.py 10000 -m src/models/model.pkl

-h: For more Info
