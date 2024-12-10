import argparse
import pickle as pkl

# Validating function to ensure the value is positive
def positive_int(value):
	
	if not value.isdigit() or int(value) < 0:
		raise argparse.ArgumentTypeError(f"\"{value}\" is an invalid value. Must be >= 0.")
	return int(value)

def main():
	# Parse the arguments
	parser = argparse.ArgumentParser(description='Predict the price of a car')
	parser.add_argument('Kilometers', type=positive_int
					 	,help='Kilometers of the car', )
	parser.add_argument('-m', '--model', help='File name of the model',
						 default='model.pkl')
	args = parser.parse_args()

	# Load the model
	try:
		with open(args.model, mode='rb') as file:
			model = pkl.load(file)
	except:
		print(f"Error: File {args.model} not found")
		exit(-1)

	# Predict the price
	price = model['theta0'] + model['theta1'] * args.Kilometers
	
	# Evaluate price
	if price < 0:
		price = 0

	print(f"Price for {args.Kilometers} km: {price}")



if __name__ == '__main__':
	main()