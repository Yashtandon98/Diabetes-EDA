import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tkinter import *

db = pd.read_csv("C://Users//summerintern.yash//Desktop//Diabetes-EDA//diabetes.csv")

y = db.Outcome
x = db.drop('Outcome', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

lr = LogisticRegression(C = 1).fit(x_train, y_train)
print('Accuracy on training set: {:.3f}'.format(lr.score(x_train, y_train)))
print('Accuracy on test set: {:.3f}'.format(lr.score(x_test, y_test)))

def predi():
    p = e1.get()
    p = int(p)
    g = e2.get()
    g = int(g)
    bp = e3.get()
    bp = int(bp)
    s = e4.get()
    s = int(s)
    i = e5.get()
    i = int(i)
    b = e6.get()
    b = float(b)
    d = e7.get()
    d = float(d)
    a = e8.get()
    a = int(a)
    P = lr.predict([[p,g,bp,s,i,b,d,a]])
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