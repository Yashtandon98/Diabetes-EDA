import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from tkinter import *

db = pd.read_csv("C://Users//summerintern.yash//Desktop//Diabetes-EDA//diabetes.csv")

x_train, x_test, y_train, y_test = train_test_split(db.loc[:,db.columns != 'Outcome'], db['Outcome'],
                                                   stratify = db['Outcome'], random_state=66)

sv = SVC()
sv.fit(x_train, y_train)

print("Accuracy on training set: {:.3f}".format(sv.score(x_train, y_train)))
print("Accuracy on test set: {:.3f}".format(sv.score(x_test, y_test)))
#Here the SVM model overfits
#This happens as all the features are not scaled
#To overcome this, the data should be rescaled so that all the features are on the same scale

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

svs = SVC(C=1000)
svs.fit(x_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(svs.score(x_train_scaled, y_train)))
print("Accuracy on test set: {:.3f}".format(svs.score(x_test_scaled, y_test)))

def predi():
    p = e1.get()
    g = e2.get()
    bp = e3.get()
    s = e4.get()
    i = e5.get()
    b = e6.get()
    d = e7.get()
    a = e8.get()
    P = svs.predict([[p,g,bp,s,i,b,d,a]])
    if(P[0] == 0):
        t.delete(1.0, 'end-1c')
        t.insert('end-1c','NOT DIABETIC')
    else:
        t.delete(1.0, 'end-1c')
        t.insert('end-1c','DIABETIC')

master = Tk()
master.title('Diabetes Predictor')
master.geometry('300x300')
Label(master, text = 'Enter the following fields:').grid(row = 0)
Label(master, text = 'Pregnancies').grid(row = 2)
Label(master, text = 'Glucose').grid(row = 3)
Label(master, text = 'Blood Pressure').grid(row = 4)
Label(master, text = 'Skin Thickness').grid(row = 5)
Label(master, text = 'Insulin').grid(row = 6)
Label(master, text = 'BMI').grid(row = 7)
Label(master, text = 'Diabetes Pedigree Function').grid(row = 8)
Label(master, text = 'Age').grid(row = 9)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)

e1.grid(row = 2, column = 1)
e2.grid(row = 3, column = 1)
e3.grid(row = 4, column = 1)
e4.grid(row = 5, column = 1)
e5.grid(row = 6, column = 1)
e6.grid(row = 7, column = 1)
e7.grid(row = 8, column = 1)
e8.grid(row = 9, column = 1)

b = Button(master, text = 'Predict', command = predi)
b.place(x=50, y=200)

t = Text(master, width = 10, height = 2)
t.grid(row = 11, column = 1)
mainloop()