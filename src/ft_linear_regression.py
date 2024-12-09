import bamboo as bb
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('../datasets/data.csv')
#print(bb.describe(df))

df_scaler = bb.StandardScaler(df)

X = df['km']
Y = df['price']

#Scaling
X_scaler = df_scaler['km']
Y_scaler = df_scaler['price']

#Regression
theta1 = sum((X_scaler - bb.mean(X_scaler)) * (Y_scaler - bb.mean(Y_scaler))) / sum((X_scaler - bb.mean(X_scaler))**2)
theta0 = bb.mean(Y_scaler) - theta1 * bb.mean(X_scaler)

#Unscaling
theta1_unscaled = theta1 * bb.std(Y) / bb.std(X)
theta0_unscaled = bb.mean(Y) - theta1_unscaled * bb.mean(X)

#Printing
print(f"Theta0: {theta0}")
print(f"Theta1: {theta1}")
print(f"cost: {pd.Series((Y_scaler - (theta0 + theta1 * X_scaler))**2).mean()}")

print(f"Price for 0 km: {theta0_unscaled + theta1_unscaled * 0}")

theta0_regress = theta0

#Gradient Descent
theta0, theta1 = 0, 0
learning_rate = 5e-1
epochs = 15

theta0_history = []
theta1_history = []
cost_history = []

print("Gradient Descent")
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

print(f"Theta0: {theta0}")
print(f"Theta1: {theta1}")

# Visualización de los resultados

# Graficar la evolución de los parámetros (theta0 y theta1)
plt.figure(figsize=(12, 6))

# Subgráfico 1: Evolución de theta0 y theta1
plt.subplot(1, 2, 1)
plt.plot(range(epochs), theta0_history, label=r'$theta_0$', color='blue')
plt.plot(range(epochs), theta1_history, label=r'$theta_1$', color='red')
plt.xlabel('Epoch')
plt.ylabel('Parameter Value')
plt.title('Evolución de $\theta_0$ y $\theta_1$ durante el Descenso de Gradiente')
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

# Guardar el modelo
model = {'theta0': theta0, 'theta1': theta1}
with open('model.pkl', mode='wb') as file:
    pkl.dump(model, file)
print("Modelo exportado a model.pkl")