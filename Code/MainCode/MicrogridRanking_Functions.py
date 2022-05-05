# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:23:48 2022

@author: ninad gaikwad and sajjad u. mahmud
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import numpy as np
import networkx as nx
from operator import itemgetter

# Custom Modules

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
    New=Abs_eigenvalue_list
    # for i in range(len(Abs_eigenvalue)): 
    #     if Abs_eigenvalue[i] <= 10e-15:
    #         Abs_eigenvalue[i] = 0
    #         New.append(Abs_eigenvalue[i])
    #     if Abs_eigenvalue[i] != 0: 
    #         New.append(Abs_eigenvalue[i])
    New.sort()
    New = np.array(New)
    Lambda2_value = float(New[1])
    Lambda2_index = Abs_eigenvalue_list.index(Lambda2_value)
    # print(Right_eigenvector)

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