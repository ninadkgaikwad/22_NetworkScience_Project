# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:21:24 2022

@author: ninad gaikwad and sajjad u. mahmud
"""
import pandas as pd
import numpy as np
import scipy.io as sio
import networkx as nx
import functions as fn

fn.banner()

# =============================================================================
# creating file name to be saved
# =============================================================================
DG_dict_file_name = 'IEEE_8500_DG' 
Comm_dict_file_name = 'IEEE_8500_Comm_P' 

probability_list = [1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# =============================================================================
# reading IEEE Line Data
# =============================================================================
excel_data = pd.read_excel (r'C:\Users\sajjaduddin.mahmud\OneDrive - Washington State University (email.wsu.edu)\CPT_S_591_Spring2022\22_NetworkScience_Project\22_NetworkScience_Project\Sajjad\IEEE Line Data\Line_Data_IEEE8500.xls')

list_A = excel_data['Node A'].tolist()
list_B = excel_data['Node B'].tolist()

# print ("List of node A =", list_A)
# print ("List of node B =", list_B)


# =============================================================================
# concatenating list A and B
# =============================================================================
list_AB= []
for i in list_A:
    list_AB.append(i)
for i in list_B:
    list_AB.append(i)
# print("Concatenated list of node A and B =", list_AB)


# =============================================================================
# creating unique list
# =============================================================================
UList_AB = fn.get_unique_list(list_AB)
# print("Unique list of node A and B =", UList_AB)



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
# print(Adj_DG)


# =============================================================================
# getting the degree for DG
# =============================================================================
degree_DG = np.sum(Adj_DG, axis=1)
# print(degree_DG)


# =============================================================================
# getting diagonal matrix for DG
# =============================================================================
degree_diag_DG = np.eye(N)
entries = np.diag_indices_from(degree_diag_DG)
degree_diag_DG[entries] = degree_DG
# print(degree_diag_DG)


# =============================================================================
# computing Laplacian matrix for DG
# =============================================================================
laplacian_DG = degree_diag_DG - Adj_DG
# print(laplacian_DG)


# =============================================================================
# saving dictionary for DG
# =============================================================================
IEEE_data_dict = {'Adj_DG':Adj_DG, 'degree_DG':degree_DG,'laplacian_DG':laplacian_DG}
sio.savemat(DG_dict_file_name,IEEE_data_dict)

# =============================================================================
# Communication network
# =============================================================================

for i in range(len(probability_list)):
    p = probability_list[i]

    # =============================================================================
    # creating communication network
    # =============================================================================
    
    Comm_network = nx.erdos_renyi_graph(N,p)
    
    
    # =============================================================================
    # creating adjacency matrix for DG
    # =============================================================================
    Adj_Comm = nx.adjacency_matrix(Comm_network)
    Adj_Comm = Adj_Comm.toarray()
    # print(Adj_Comm)
    
    
    # =============================================================================
    # getting the degree for Comm
    # =============================================================================
    degree_Comm = np.sum(Adj_Comm, axis=1)
    # print(degree_Comm)
    
    
    # =============================================================================
    # getting diagonal matrix for Comm
    # =============================================================================
    degree_diag_Comm = np.eye(N)
    entries = np.diag_indices_from(degree_diag_Comm)
    degree_diag_Comm[entries] = degree_Comm
    # print(degree_diag_Comm)
    
    
    # =============================================================================
    # computing Laplacian matrix for comm
    # =============================================================================
    laplacian_Comm = degree_diag_Comm - Adj_Comm
    # print(laplacian_Comm)
    
    
    # =============================================================================
    # saving dictionary for comm
    # =============================================================================
    Comm_data_dict = {'Adj_Comm':Adj_Comm, 'degree_Comm':degree_Comm,'laplacian_Comm':laplacian_Comm}
    file_name_current = Comm_dict_file_name + str(p*100)
    sio.savemat(file_name_current,Comm_data_dict)
