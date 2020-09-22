import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random


inp_df = pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\training_feature_matrix.xlsx", header=None).to_numpy()
y= pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\training_output.xlsx",header=None).to_numpy()

def cost_least_angle(X,y,theta,lamb=100):
    m=len(y)
    predict= np.dot(X,theta)
    cost1 =  np.sum(np.square(predict-y))+ lamb*np.sum(abs(theta))
    return cost1/2
def cost(X,y,theta):
    predict = np.dot(X, theta)
    cost1 = np.sum(np.square(predict - y))
    return cost1 / 2
def cost_ridge(X,y,theta,lamb=100):
    m=len(y)
    predict= np.dot(X,theta)
    cost1 =  np.sum(np.square(predict-y))+ lamb*np.sum(np.square(theta))
    return cost1/2


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

def ridge_grad_descent(X , y, n_iter, alpha,theta,lamb=10):
    m=len(y)
    cost_history = np.zeros(n_iter)
    theta_hist= np.zeros((n_iter,3))
    for iter1 in range(n_iter) :
        predict= np.dot(X,theta)
        theta =(1-alpha*lamb)* theta -  alpha*(X.T.dot((predict-y)))
        theta_hist[iter1 , :]= theta.T
        cost_history[iter1]= cost_ridge(X,y,theta)
    return theta, cost_history, theta_hist

def least_angle_grad_descent(X , y, n_iter, alpha,theta,lamb=10):
    m=len(y)
    cost_history = np.zeros(n_iter)
    theta_hist= np.zeros((n_iter,3))
    for iter1 in range(n_iter) :
        predict= np.dot(X,theta)
        theta =(theta-alpha*lamb*np.sign(theta)) -  alpha*(X.T.dot((predict-y)))
        theta_hist[iter1 , :]= theta.T
        cost_history[iter1]= cost_least_angle(X,y,theta)
    return theta, cost_history, theta_hist

m=y.shape[0]
X0= np.ones((m,1))
Xnew=np.hstack((X0,inp_df))
ncol= Xnew.shape[1]   # cal. no. of cols

alpha = 0.0001
theta= np.random.randn(ncol,1)
n_iter= 100

ym = np.mean(y)
ystd = np.std(y)
Xm1 = np.mean(Xnew[:,1])
Xstd1 = np.std(Xnew[:,1])
Xm2 = np.mean(Xnew[:,2])
Xstd2 = np.std(Xnew[:,2])

Xnew[:,1] = (Xnew[:,1]-np.mean(Xnew[:,1]))/np.std(Xnew[:,1])
Xnew[:,2] = (Xnew[:,2]-np.mean(Xnew[:,2]))/np.std(Xnew[:,2])
y=(y-ym)/ystd

n_iter=200
theta , cost_history , theta_hist = grad_descent(Xnew , y , n_iter, alpha, theta)

plt.plot(cost_history)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

n_iter=500
theta_ridge=np.random.randn(ncol,1)
theta_ridge , cost_history_ridge,theta_hist_ridge = ridge_grad_descent(Xnew , y , n_iter, alpha, theta_ridge)
plt.plot(cost_history_ridge)
plt.show()

n_iter=200
theta_least_angle=np.random.randn(ncol,1)
theta_least_angle , cost_history_least_angle , theta_hist_least_angle = least_angle_grad_descent(Xnew , y , n_iter, alpha, theta_least_angle)
plt.plot(cost_history_least_angle)
plt.show()

out_df = pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\test_feature_matrix.xlsx", header=None).to_numpy()
y1= pd.read_excel(r"D:\3-1\course\Neural Networks\assignment1\test_output.xlsx",header=None).to_numpy()

m1=y1.shape[0]
X= np.ones((m1,1))
Xt=np.hstack((X,out_df))

Xt[:,1] = (Xt[:,1]-Xm1)/Xstd1  #normalize all X and y
Xt[:,2] = (Xt[:,2]-Xm2)/Xstd2
y1=(y1-ym)/ystd

print(cost(Xt,y1,theta)/m)
print(cost_ridge(Xt,y1,theta_ridge)/m)
print(cost_least_angle(Xt,y1,theta_least_angle)/m)
