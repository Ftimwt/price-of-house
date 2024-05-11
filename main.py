import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# get dataset
data = pd.read_csv('priceandsize.csv')
data.head()

# Extract features (size) and target variable (price)
X = data['size'].values.reshape(-1, 1)
y = data['price'].values

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Function to predict price based on size
def predict_price(size):
    price = model.predict(np.array(size).reshape(-1, 1))
    return price[0]

# Main function
def main():
    # User input for size
    size_input = float(input("Enter the size of your house (Square meters): "))
    
    # Predict price
    predicted_price = predict_price(size_input)
    print(f"Predicted price for your house with size {size_input} Square meters: ${predicted_price:.2f}")

    # Visualize the dataset and regression line
    plt.scatter(X, y, color='green', label='Original data')
    plt.plot(X, model.predict(X), color='yellow', label='Linear regression')
    plt.xlabel('Size (Square meters)')
    plt.ylabel('Price ($)')
    plt.title('Predict House Price')
    plt.show()
    plt.legend()
    plt.grid(True)
    

if __name__ == "__main__":
    main()