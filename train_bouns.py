import csv
import matplotlib
matplotlib.use('Agg') # Set the backend to non-interactive to avoid the warning
import matplotlib.pyplot as plt

def train_bonus():
    mileage, price = [], []
    try:
        with open('data.csv', mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mileage.append(float(row['km']))
                price.append(float(row['price']))
    except FileNotFoundError:
        print("Error: data.csv not found.")
        return

    m = len(mileage)
    theta0, theta1 = 0.0, 0.0
    learning_rate = 0.1
    epochs = 1000

    # Normalization (Min-Max Scaling) is required for gradient descent to converge
    min_m, max_m = min(mileage), max(mileage)
    norm_mileage = [(x - min_m) / (max_m - min_m) for x in mileage]

    for _ in range(epochs):
        sum0, sum1 = 0.0, 0.0
        for i in range(m):
            # Hypothesis: estimatePrice(mileage) = theta0 + (theta1 * mileage) [cite: 80]
            prediction = theta0 + (theta1 * norm_mileage[i])
            error = prediction - price[i]
            sum0 += error
            sum1 += error * norm_mileage[i]
        
        # Simultaneous update using the specified formulas [cite: 86, 89]
        theta0 -= learning_rate * (1/m) * sum0
        theta1 -= learning_rate * (1/m) * sum1

    # Rescale thetas for original data for use in the prediction program [cite: 84]
    final_theta1 = theta1 / (max_m - min_m)
    final_theta0 = theta0 - (theta1 * min_m / (max_m - min_m))

    with open('theta.csv', 'w') as f:
        f.write(f"{final_theta0},{final_theta1}")

    # BONUS: Plotting data distribution and regression line 
    plt.figure(figsize=(10, 6))
    plt.scatter(mileage, price, color='blue', label='Data Points (Distribution)')
    
    line_x = [min(mileage), max(mileage)]
    line_y = [final_theta0 + final_theta1 * x for x in line_x]
    plt.plot(line_x, line_y, color='red', linewidth=2, label='Linear Regression Line')
    
    plt.xlabel('Mileage (km)')
    plt.ylabel('Price')
    plt.title('Car Price vs Mileage: Linear Regression Result')
    plt.legend()
    plt.grid(True)
    
    # Save the file instead of showing it
    plt.savefig('regression_plot.png')
    print("Training complete. Parameters saved to 'theta.csv'. Plot saved as 'regression_plot.png'.")

if __name__ == "__main__":
    train_bonus()