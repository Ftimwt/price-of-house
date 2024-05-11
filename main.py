import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from predict import Predict

def integer_input(msg: str) -> int:
    ### Just input integer values not string or anything else
    while True:
        try:
            return int(input(msg))
        except:
            print("Enter valid integer value !")

# Main function
def main():
    # create new instance from `Predict` object
    predict = Predict(file_name='priceandsize.csv')
    while True:
        print("[Option 1] Draw a plot from predicts")
        print("[Option 2] predict house price with size (in square meter)")
        print("[Option Q] exit from application")
        prompt = input("Choose option: ").lower()
        if prompt == "1":
            predict.draw_plot()
        elif prompt == "2":
            size = integer_input("Enter house size in meter square: ")
            price = predict.predict_price(size)
            print(f"\nPredict price for this house with {size} meters square is {price:.2f} Million Toman\n")
        elif prompt == "q":
            print("Goodbye...")
            exit()




if __name__ == "__main__":
    main()