from tkinter import *
import turtle
from tkinter import Tk, Label, Button
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
import tensorflow as tf

class SimpleInterestCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title('Simple Interest Calculator')
        self.root.geometry("500x400")
        self.root.configure(bg='grey')

        # Tensor Example
        self.tensor_operations_example()

        self.create_widgets()

    def calculate_interest(self):
        p = float(self.principle.get())
        r = float(self.rate.get())
        t = float(self.time.get())
        i = (p * r * t) / 100
        interest = round(i, 2)

        self.result.destroy()

        message = Label(
            self.result_frame,
            text=f"Interest on Rs.{p} at rate of interest {r}% for {t} years is Rs.{interest}",
            bg="grey",
            font=("Calibri", 12),
            width=55
        )
        message.place(x=20, y=40)
        message.pack()

    def tensor_operations_example(self):
        # Example tensor operations with TensorFlow
        tensor_example = tf.constant([[1, 2], [3, 4]])
        tensor_squared = tf.square(tensor_example)
        print("Tensor Squared:")
        print(tensor_squared)

    def turtle_graphics_example(self):
        # Example turtle graphics
        turtle.forward(100)
        turtle.right(90)
        turtle.forward(100)
        turtle.done()

    def gui_example(self):
        # Example GUI using tkinter
        root = Tk()
        label = Label(root, text="Hello, GUI!")
        button = Button(root, text="Click me")
        label.pack()
        button.pack()
        root.mainloop()

    def sklearn_example(self):
        # Example machine learning with sklearn
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

    def keras_example(self):
        # Example deep learning with Keras
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=8))
        model.add(Dense(units=1, activation='sigmoid'))

    def data_mining_example(self):
        # Example data mining with pandas
        data = {'Name': ['John', 'Jane', 'Bob'], 'Age': [28, 35, 22]}
        df = pd.DataFrame(data)

    def data_processing_example(self):
        # Example data processing with numpy
        array = np.array([[1, 2, 3], [4, 5, 6]])
        sum_result = np.sum(array)
        print("Sum of the array:", sum_result)

    def create_widgets(self):
        app_label = Label(
            self.root,
            text="SIMPLE INTEREST CALCULATOR",
            fg="black",
            bg="grey",
            font=("Calibri", 20),
            bd=5
        )
        app_label.place(x=20, y=20)

        principle_label = Label(
            self.root,
            text="Principle in Rs",
            fg="black",
            bg="grey",
            font=("Calibri", 12),
            bd=1
        )
        principle_label.place(x=20, y=92)

        self.principle = Entry(self.root, text="", bd=2, width=22)
        self.principle.place(x=200, y=92)

        rate_label = Label(
            self.root,
            text="Rate of Interest in %",
            fg="black",
            bg="grey",
            font=("Calibri", 12)
        )
        rate_label.place(x=20, y=140)

        self.rate = Entry(self.root, text="", bd=2, width=15)
        self.rate.place(x=200, y=142)

        time_label = Label(
            self.root,
            text="Time in Yrs",
            fg="black",
            bg="grey",
            font=("Calibri", 12)
        )
        time_label.place(x=20, y=185)

        self.time = Entry(self.root, text="", bd=2, width=15)
        self.time.place(x=200, y=187)

        calculate_button = Button(
            self.root,
            text="CALCULATE",
            fg="black",
            bg="grey",
            bd=4,
            command=self.calculate_interest
        )
        calculate_button.place(x=20, y=250)

        self.result_frame = LabelFrame(
            self.root,
            text="Result",
            bg="grey",
            font=("Calibri", 12)
        )
        self.result_frame.pack(padx=20, pady=20)
        self.result_frame.place(x=20, y=300)

        self.result = Label(
            self.result_frame,
            text="Your result will be displayed here",
            bg="grey",
            font=("Calibri", 12),
            width=55
        )
        self.result.place(x=20, y=20)
        self.result.pack()

if __name__ == "__main__":
    root = Tk()
    app = SimpleInterestCalculator(root)
    root.mainloop()
