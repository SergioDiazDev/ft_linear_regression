import bamboo as bb
import pandas as pd

# Load the data
df = pd.read_csv('../datasets/data.csv')
#print(bb.describe(df))

df_scaler = bb.StandardScaler(df)

#Variables
theta0, theta1 = 0, 0
learning_rate = 0.5
epochs = 200

X = df['km']
Y = df['price']

X_scaler = df_scaler['km']
Y_scaler = df_scaler['price']

theta1 = sum((X_scaler - bb.mean(X_scaler)) * (Y_scaler - bb.mean(Y_scaler))) / sum((X_scaler - bb.mean(X_scaler))**2)
theta0 = bb.mean(Y_scaler) - theta1 * bb.mean(X_scaler)

#print(df_scaler)

theta1_unscaled = theta1 * bb.std(Y) / bb.std(X)
theta0_unscaled = bb.mean(Y) - theta1_unscaled * bb.mean(X)

print(f"Theta0: {theta0}")
print(f"Theta1: {theta1}")

print(f"Price for 0 km: {theta0_unscaled + theta1_unscaled * 0}")