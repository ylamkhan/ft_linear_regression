def predict():
    # Initialize thetas to 0 [cite: 81]
    theta0, theta1 = 0.0, 0.0
    
    # Load trained parameters if they exist [cite: 84]
    try:
        with open('theta.csv', 'r') as f:
            content = f.read().split(',')
            theta0, theta1 = float(content[0]), float(content[1])
    except FileNotFoundError:
        print("Note: Model not trained yet. Using default thetas (0).")

    try:
        mileage = float(input("Please enter a mileage: ")) # [cite: 78]
        # Linear hypothesis [cite: 80]
        estimate_price = theta0 + (theta1 * mileage)
        print(f"Estimated price: {estimate_price:.2f}")
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    predict()