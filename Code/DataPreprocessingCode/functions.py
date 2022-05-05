# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:50:58 2022

@author: sajjaduddin.mahmud
"""
import pandas as pd
import numpy as np
import networkx as nx
from scipy import linalg
from operator import itemgetter
# from termcolor import colored

# =============================================================================
# Banner
# =============================================================================
def banner():
    stars  ='**************************************************************************************'
    title  ='*   Application of Network Science to Power System Control Algorithm Vulnerability   *'
    author ='*                      Ninad Gaikwad and Sajjad Uddin Mahmud                         *' 
    print(colored(stars,color='green'))
    print(colored(title,color='white'))
    print(colored(author,color='white'))
    print(colored(stars,color='green'))
    
    
# =============================================================================
# creating unique list
# =============================================================================
def get_unique_list(Numbers):
    Unique_numbers = set(Numbers) # converting list into set
    Unique_numbers = list(Unique_numbers)
    Unique_numbers.sort()
    
    return Unique_numbers



# =============================================================================
# Creating a diagonal matrix
# =============================================================================
def Get_diagonal_mat(Dim,Diag_values):
    Diagonal_mat = np.eye(Dim)
    Entries = np.diag_indices_from(Diagonal_mat)
    Diagonal_mat[Entries] = Diag_values
    return Diagonal_mat 



# =============================================================================
# Computing linear system matrix
# =============================================================================
def Compute_linearsystem_matrix(N,l,z,k,m,B,Lp,Ld):
    
    # Creating K matrix
    Dim_K = N
    Diag_values_k = k
    K = Get_diagonal_mat(Dim_K, Diag_values_k)
        
    
    # Creating I_l_N matrix
    I_l_l = np.eye(l)
    I_l_N = np.block([[I_l_l,np.zeros((l,N-l))],[np.zeros((N-l,l)),np.zeros((N-l,N-l))]])
        

    # Creating I_l_Nl matrix
    I_l_Nl = np.block([[np.zeros((l,N-l))],[np.eye(N-l)]])
        
 
    # Creating I_Nl_N matrix
    I_Nl_N = np.block([np.zeros((N-l,l)),np.eye(N-l)])
        
        
    # Creating Z matrix
    Dim_Z = N-l
    Diag_values_Z = z
    Z = Get_diagonal_mat(Dim_Z, Diag_values_Z)
        
    
    # Creating M inverse matrix
    Dim_M = N-l
    Diag_values_M = m
    M = Get_diagonal_mat(Dim_M, Diag_values_M)
    M_inv = np.block([np.linalg.inv(M)])
        
    
    # Creating close-loop matrix
    X = Lp + np.matmul(K,Ld) + B
    A11 = - np.matmul(I_l_N,X)
    A12 = I_l_Nl
    A21 = - np.matmul(M_inv,np.matmul(I_Nl_N,X))
    A22 = - Z
    
    A = np.block([[A11,A12],[A21,A22]])
        
    
    # Computing eigenvalues and eigevectors of close-loop matrix
    Eig_val_A, Eig_LVec_A, Eig_RVec_A = linalg.eig(A,left=True,right=True)

    return A, Eig_val_A, Eig_LVec_A, Eig_RVec_A


# =============================================================================
# Computing ranking with degree centrality
# =============================================================================
def Get_ranking_degree(Adjacency_matrix):
    Adj_mat = Adjacency_matrix
    Graph = nx.from_numpy_matrix(Adj_mat)
    Rank_dict = nx.degree_centrality(Graph)
    Rank_list = list(Rank_dict.items()) # converting dictionary into list
    Rank_list_sort = sorted(Rank_list, key=itemgetter(1), reverse=True) # sorting list as per Rank in descending order
    Rank_array = np.array(Rank_list_sort) # converting list into nd.array
    
    return Rank_array


# =============================================================================
# Computing ranking with pagerank
# =============================================================================
def Get_ranking_pagerank(Adjacency_matrix):
    Adj_mat = Adjacency_matrix
    Graph = nx.from_numpy_matrix(Adj_mat)
    Rank_dict = nx.pagerank(Graph)
    Rank_list = list(Rank_dict.items()) # converting dictionary into list
    Rank_list_sort = sorted(Rank_list, key=itemgetter(1), reverse=True) # sorting list as per Rank in descending order
    Rank_array = np.array(Rank_list_sort) # converting list into nd.array
    
    return Rank_array


# =============================================================================
# Computing ranking with eigenanalysis
# =============================================================================
def Get_ranking_eigenanalysis(Adjacency_matrix, Laplacian_matrix):
    
    # Reading adjacency and laplacian matrix
    Adj_mat = Adjacency_matrix
    Lap_mat = Laplacian_matrix
     

    # Getting eigenvalue, right eigenvector, absolute real part of eigenvalue
    Eigenvalue, Right_eigenvector = np.linalg.eig(Lap_mat)
    Real_eigenvalue_list = Eigenvalue.real
    Real_eigenvalue = np.array(Real_eigenvalue_list) # converting list to nd.array
    Abs_eigenvalue = np.abs(Real_eigenvalue) # getting |real eigenvalue|
    Abs_eigenvalue_list = Abs_eigenvalue.tolist()
     
     
    # Getting Lambda2 - the |real eigenvalue| closest to zero
    New=[]
    for i in range(len(Abs_eigenvalue)): 
        if Abs_eigenvalue[i] <= 10e-15:
            Abs_eigenvalue[i] = 0
        if Abs_eigenvalue[i] != 0: 
            New.append(Abs_eigenvalue[i])
    New.sort()
    New = np.array(New)
    Lambda2_value = float(New[:1])
    Lambda2_index = Abs_eigenvalue_list.index(Lambda2_value)
    print(Right_eigenvector)

    # Getting eigenvector of Lambda2
    U = Right_eigenvector[:,Lambda2_index]


    # Computing lambda2 deviations
    Total_node = np.prod(U.shape)
    Lambda2_deviation = np.zeros((Total_node,))
    for i in range(Total_node):
                
        # Getting adjacency matrix row corresponding to current node
        Current_adjacency_matrix_row = Adj_mat[i,:]
        
        # Finding set N_i
        Set_N_i = []
        for j in range(Total_node):
            if Current_adjacency_matrix_row[j] == 1:
                Set_N_i.append(j)
        
        
        for j in range(len(Set_N_i)):
            Lambda2_deviation[i] = Lambda2_deviation[i] + U[j]*(U[i]-U[j])
        
        Lambda2_deviation[i] = Lambda2_deviation[i] / (1-U[i]**2) 
        
        Node_number = np.arange(Total_node)
        Lambda2_deviation_ranklist = np.block([np.reshape(Node_number,(Total_node,1)),np.reshape(Lambda2_deviation,(Total_node,1))])
        

    # Ranking with Lambda2_deviation
    Rank_list = Lambda2_deviation_ranklist.tolist()
    Rank_list_sort = sorted(Rank_list, key=itemgetter(1), reverse=True) # sorting list as per Rank in descending order
    Rank_array = np.array(Rank_list_sort) # converting list into nd.array



    return Rank_array
