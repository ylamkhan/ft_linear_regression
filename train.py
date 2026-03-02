import csv

def train():
    # Load data [cite: 83, 106]
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

    m = len(mileage) # [cite: 87]
    theta0, theta1 = 0.0, 0.0 # Initial values [cite: 81]
    learning_rate = 0.1
    epochs = 1000

    # Normalization for stability
    min_m, max_m = min(mileage), max(mileage)
    norm_mileage = [(x - min_m) / (max_m - min_m) for x in mileage]

    # Gradient Descent loop [cite: 63, 86, 89]
    for _ in range(epochs):
        sum0, sum1 = 0.0, 0.0
        for i in range(m):
            # Hypothesis: estimatePrice = theta0 + (theta1 * mileage) [cite: 80, 88]
            prediction = theta0 + (theta1 * norm_mileage[i])
            error = prediction - price[i]
            sum0 += error
            sum1 += error * norm_mileage[i]
        
        # Simultaneous update 
        tmp_theta0 = learning_rate * (1/m) * sum0
        tmp_theta1 = learning_rate * (1/m) * sum1
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    # Rescale theta for original mileage values
    final_theta1 = theta1 / (max_m - min_m)
    final_theta0 = theta0 - (theta1 * min_m / (max_m - min_m))

    # Save variables for the prediction program [cite: 84]
    with open('theta.csv', 'w') as f:
        f.write(f"{final_theta0},{final_theta1}")
    print("Training complete. Parameters saved.")

if __name__ == "__main__":
    train()