# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:21:24 2022

@author: sajjaduddin.mahmud
"""
import pandas as pd
import numpy as np
import scipy.io as sio
import functions as fn

fn.banner()

# =============================================================================
# creating file name to be saved
# =============================================================================
file_name = 'IEEE_13_DG'  


# =============================================================================
# reading IEEE Line Data
# =============================================================================
excel_data = pd.read_excel (r'C:\Users\sajjaduddin.mahmud\OneDrive - Washington State University (email.wsu.edu)\CPT_S_591_Spring2022\22_NetworkScience_Project\22_NetworkScience_Project\Sajjad\IEEE Line Data\Line_Data_IEEE13.xls')

list_A = excel_data['Node A'].tolist()
list_B = excel_data['Node B'].tolist()

print ("List of node A =", list_A)
print ("List of node B =", list_B)


# =============================================================================
# concatenating list A and B
# =============================================================================
list_AB= []
for i in list_A:
    list_AB.append(i)
for i in list_B:
    list_AB.append(i)
print(list_AB)


# =============================================================================
# creating unique list
# =============================================================================
UList_AB = fn.get_unique_list(list_AB)
print("Unique list of node A and B =", UList_AB)



# =============================================================================
# creating adjacency matrix for DG
# =============================================================================
N = len(UList_AB)
Adj_DG = np.zeros((N,N))
#print(Adj_DG)

count = 0
for i in range(len(list_A)):
    xA = list_A[i]
    xB = list_B[i]
    xA_index = UList_AB.index(xA)
    xB_index = UList_AB.index(xB)
    Adj_DG[xA_index,xB_index] = 1
    Adj_DG[xB_index,xA_index] = 1
print(Adj_DG)


# =============================================================================
# getting the degree
# =============================================================================
degree_DG = np.sum(Adj_DG, axis=1)
print(degree_DG)


# =============================================================================
# getting diagonal matrix
# =============================================================================
degree_diag_DG = np.eye(N)
entries = np.diag_indices_from(degree_diag_DG)
degree_diag_DG[entries] = degree_DG
print(degree_diag_DG)


# =============================================================================
# computing Laplacian matrix
# =============================================================================
laplacian_DG = degree_diag_DG - Adj_DG
print(laplacian_DG)


# =============================================================================
# saving dictionary
# =============================================================================
IEEE_data_dict = {'Adj_DG':Adj_DG, 'degree_DG':degree_DG,'laplacian_DG':laplacian_DG}
sio.savemat(file_name,IEEE_data_dict)
    