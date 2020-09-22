import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

inp_df = pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\training_feature_matrix.xlsx", header=None).to_numpy()
y= pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\training_output.xlsx",header=None).to_numpy()

def mse(X,y,theta):
    m=len(y)
    predict= np.dot(X,theta)
    cost1 = np.sum(np.square(predict-y))/m
    return cost1

def cost(X,y,theta):
    predict = np.dot(X, theta)
    cost1 = np.sum(np.square(predict - y))
    return cost1 / 2

def grad_descent(X , y, n_iter, alpha,theta):
    m=len(y)
    cost_history = np.zeros(n_iter)
    theta_hist= np.zeros((n_iter,3))
    for iter1 in range(n_iter) :
        predict= np.dot(X,theta)
        theta = theta -  alpha*(X.T.dot((predict-y)))
        theta_hist[iter1 , :]= theta.T
        cost_history[iter1]= cost(X,y,theta)
    return theta, cost_history, theta_hist

m=y.shape[0]
X0= np.ones((m,1))
Xnew=np.hstack((X0,inp_df))
ncol= Xnew.shape[1]   # cal. no. of cols

alpha = 0.0001
theta= np.random.randn(ncol,1)
n_iter= 300

xstd1=np.std(Xnew[:,1])
xstd2=np.std(Xnew[:,2])
xmean1=np.mean(Xnew[:,1])
xmean2=np.mean(Xnew[:,2])
ymean=np.mean(y)
ystd=np.std(y)

Xnew[:,1] = (Xnew[:,1]-np.mean(Xnew[:,1]))/np.std(Xnew[:,1])
Xnew[:,2] = (Xnew[:,2]-np.mean(Xnew[:,2]))/np.std(Xnew[:,2])
y=(y-np.mean(y))/np.std(y)

theta , cost_history , theta_hist = grad_descent(Xnew , y , n_iter, alpha, theta)

out_df = pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\test_feature_matrix.xlsx", header=None).to_numpy()
y1= pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\test_output.xlsx",header=None).to_numpy()

m1=y1.shape[0]
X= np.ones((m1,1))
Xt=np.hstack((X,out_df))

Xt[:,1] = (Xt[:,1]-xmean1)/xstd1  #normalize all X and y
Xt[:,2] = (Xt[:,2]-xmean2)/xstd2
y1=(y1-ymean)/ystd


print(theta)
print(mse(Xt,y1,theta))

plt.plot(cost_history)
plt.show()
