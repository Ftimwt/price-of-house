from os import path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class Predict:

  def __init__(self, file_name: str) -> None:
    # check entered file was exists or not
    if path.exists(file_name):
      assert "file does not exists !"
    self._file_name = file_name

    data = pd.read_csv(file_name)
    data.head()


    # Extract size and prices from csv file
    self.x = data['size'].values.reshape(-1, 1) 
    self.y = data['price'].values

    # Create and fit the linear regression model
    self.model = LinearRegression()
    self.model.fit(self.x, self.y)

  def predict_price(self, size: float|int) -> int:
    price = self.model.predict(np.array(size).reshape(-1, 1))
    return price[0]

  def draw_plot(self):
    plt.scatter(self.x, self.y, color='purple', label='Original data')
    plt.plot(self.x, self.model.predict(self.x), color='pink', label='Linear regression')
    plt.xlabel('Size (Square meters)')
    plt.ylabel('Price (Milion Toman)')
    plt.title('Predict House Price')
    plt.show()
    plt.legend()
    plt.grid(True)