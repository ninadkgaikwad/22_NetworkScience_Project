# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:10:23 2022

@author: sajjaduddin.mahmud
"""

# Modules
import os
import scipy.io as sio
import scipy.signal as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

## Loading Linear System Microgrid Closed Loop System Matrix
A_Dict = sio.loadmat('A_Mat.mat')

A = A_Dict['A']

## Initialization
time_vector = np.arange(0,120,0.001)

Deg_Deviation = 0.5

High_x_ini = 2*np.pi*(Deg_Deviation/360)

Low_x_ini = -2*np.pi*(Deg_Deviation/360)

## Initial Condition
r,c = A.shape

Time_Len = len(time_vector)

x_ini = np.random.uniform(Low_x_ini,High_x_ini,(r,))

u = np.zeros((Time_Len,r))

## Creating Linear System

A_Sys = A

B_Sys = np.zeros((r,r))

C_Sys = np.eye(r)

D_Sys = np.zeros((r,r))

System = (A_Sys, B_Sys, C_Sys, D_Sys)

## Simulating System

tout, y, x = ss.lsim(System, u, time_vector, x_ini)

## Plotting


for i in range(r):
    
    plt.plot(time_vector, y[:,i])
    
