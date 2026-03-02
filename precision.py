import csv

def calculate_precision():
    mileage, price = [], []
    try:
        with open('data.csv', mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mileage.append(float(row['km']))
                price.append(float(row['price']))
        
        with open('theta.csv', 'r') as f:
            content = f.read().split(',')
            t0, t1 = float(content[0]), float(content[1])
    except FileNotFoundError:
        print("Error: Ensure data.csv and theta.csv exist.")
        return

    m = len(mileage)
    mean_price = sum(price) / m
    ss_res = 0  # Sum of Squares Residuals
    ss_tot = 0  # Total Sum of Squares
    mae = 0     # Mean Absolute Error

    for i in range(m):
        prediction = t0 + (t1 * mileage[i])
        ss_res += (price[i] - prediction)**2
        ss_tot += (price[i] - mean_price)**2
        mae += abs(price[i] - prediction)

    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"--- Algorithm Precision ---")
    print(f"Mean Absolute Error: {mae/m:.2f}")
    print(f"R-squared Score: {r_squared:.4f} (Closer to 1.0 is better)")

if __name__ == "__main__":
    calculate_precision()