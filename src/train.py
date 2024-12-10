# Training the model
import argparse
import pandas as pd
import bamboo as bb
import matplotlib.pyplot as plt
import pickle as pkl

def linear_regression(X, Y, X_scaler, Y_scaler):
	#Regression
	theta1_scaled = sum((X_scaler - bb.mean(X_scaler)) * (Y_scaler - bb.mean(Y_scaler))) / sum((X_scaler - bb.mean(X_scaler))**2)
	theta0_scaled = bb.mean(Y_scaler) - theta1_scaled * bb.mean(X_scaler)

	#Unscaling
	theta1_unscaled = theta1_scaled * bb.std(Y) / bb.std(X)
	theta0_unscaled = bb.mean(Y) - theta1_unscaled * bb.mean(X)

	#Printing scaled values
	print("Scaled values")
	print(f"Theta0: {theta0_scaled}")
	print(f"Theta1: {theta1_scaled}")
	print(f"cost: {pd.Series((Y_scaler - (theta0_scaled + theta1_scaled * X_scaler))**2).mean()}")
	print("---------------")

	#Printing
	print("Unscaled values")
	print(f"Theta0: {theta0_unscaled}")
	print(f"Theta1: {theta1_unscaled}")
	print(f"cost: {pd.Series((Y - (theta0_unscaled + theta1_unscaled * X))**2).mean()}")
	print("---------------")
	
def check_dataset(df):

	# Check if the dataframe is empty
	if df.empty:
		print("Error: Empty dataframe")
		exit(-1)

	# Check if the columns exist
	if 'km' not in df or 'price' not in df:
		print("Error: Columns 'km' and 'price' must exist")
		exit(-1)

	#check if columns are numeric
	if not bb.is_numeric(df['km']) or not bb.is_numeric(df['price']):
		print("Error: Columns 'km' and 'price' must be numeric")
		exit(-1)
	
	#check if std is 0
	if bb.std(df['km']) == 0 or bb.std(df['price']) == 0:
		print("Error: Standard deviation of columns 'km' and 'price' must be different from 0")
		exit(-1)

def gradient_descent_step(X, Y, X_scaler, Y_scaler, epochs, learning_rate, graph, model_name):
	#Gradient Descent
	theta0, theta1 = 0, 0

	theta0_history = []
	theta1_history = []
	cost_history = []

	print(f"Gradient Descent: epochs={epochs}, learning_rate={learning_rate}")
	print("---------------")
	for i in range(epochs):
		theta0, theta1 = bb.gradient_descent_step(theta0, theta1, X_scaler, Y_scaler, learning_rate)
		
		# Calcular la predicción
		Y_pred = theta0 + theta1 * X_scaler
		
		# Calcular el coste (error cuadrático medio)
		cost = (1/(2*len(X_scaler))) * sum((Y_scaler - Y_pred) ** 2)
		
		print(f"Epoch {i+1}: theta0={theta0}, theta1={theta1} - Cost={cost}")


		
		# Guardamos los parámetros y el coste en cada época
		theta0_history.append(theta0)
		theta1_history.append(theta1)
		cost_history.append(cost)
		
	print("---------------")
	theta1_unscaled = theta1 * bb.std(Y) / bb.std(X)
	theta0_unscaled = bb.mean(Y) - theta1_unscaled * bb.mean(X)

	#Printing Scaled values
	print("Scaled values")
	print(f"Theta0: {theta0}")
	print(f"Theta1: {theta1}")
	print(f"cost: {pd.Series((Y_scaler - (theta0 + theta1 * X_scaler))**2).mean()}")
	print("---------------")
	#Printing Unscaled values
	print("Unscaled values")
	print(f"Theta0: {theta0_unscaled}")
	print(f"Theta1: {theta1_unscaled}")
	print(f"cost: {pd.Series((Y - (theta0_unscaled + theta1_unscaled * X))**2).mean()}")
	print("---------------")

	#Export the model
	model = {'theta0': theta0_unscaled, 'theta1': theta1_unscaled}
	with open(model_name, mode='wb') as file:
		pkl.dump(model, file)
	print(f"Model exported to {model_name}")
	if graph:
		# Visualización de los resultados
		# Graficar la evolución de los parámetros (theta0 y theta1)
		plt.figure(figsize=(12, 6))

		# Subgráfico 1: Evolución de theta0 y theta1
		plt.subplot(1, 2, 1)
		plt.plot(range(epochs), theta0_history, label=r'theta_0', color='blue')
		plt.plot(range(epochs), theta1_history, label=r'theta_1', color='red')
		plt.xlabel('Epoch')
		plt.ylabel('Parameter Value')
		plt.title('Evolución de theta_0 y theta_1 durante el Descenso de Gradiente')
		plt.legend()

		# Subgráfico 2: Evolución de la función de coste
		plt.subplot(1, 2, 2)
		plt.plot(range(epochs), cost_history, color='green')
		plt.xlabel('Epoch')
		plt.ylabel('Cost')
		plt.title('Evolución de la Función de Coste')
		plt.tight_layout()

		# Mostrar los gráficos
		plt.show()

def main():
	# Parse the arguments
	parser = argparse.ArgumentParser(description='Train a linear regression model')
	parser.add_argument('datafile', help='File name of the csv data',
						nargs='?', default='../datasets/data.csv')
	parser.add_argument('-c', '--compare', action='store_true', default=False,
						help='Compare Regression with degradient descent')
	parser.add_argument('-e', '--epochs', type=int, default=15,
						help='Number of epochs for the gradient descent')
	parser.add_argument('-l', '--learning_rate', type=float, default=5e-1,
						help='Learning rate for the gradient descent')
	parser.add_argument('-g', '--graph', action='store_true', default=False, 
					 	help='Show the graph')
	parser.add_argument('-m', '--model', help='File name of the model',
						 default='model.pkl')
	args = parser.parse_args()

	# Load the data
	try:
		df = pd.read_csv(args.datafile)
	except:
		print(f"Error: File {args.datafile} not found")
		exit(-1)

	# Check the dataset
	check_dataset(df)

	# Extract the columns
	X = df['km']
	Y = df['price']

	# Scaling the data with the StandardScaler
	df_scaler = bb.StandardScaler(df)

	# Extract the columns scaled
	X_scaler = df_scaler['km']
	Y_scaler = df_scaler['price']

	#Print linear regression
	if args.compare:
		print("Linear Regression")
		linear_regression(X, Y, X_scaler, Y_scaler)
		print("---------------")

	#Regression with degradient descent
	gradient_descent_step(X, Y, X_scaler, Y_scaler, args.epochs, args.learning_rate, args.graph, args.model)
	


if __name__ == '__main__':
	main()