# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:12:03 2022

@author: ninad gaikwad and sajjad u. mahmud
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import numpy as np
# from termcolor import colored

# =============================================================================
# Banner
# =============================================================================
# def Banner():
#     Stars  ='**************************************************************************************'
#     Title  ='*   Application of Network Science to Power System Control Algorithm Vulnerability   *'
#     Author ='*                      Ninad Gaikwad and Sajjad Uddin Mahmud                         *' 
#     print(colored(Stars,color='green'))
#     print(colored(Title,color='white'))
#     print(colored(Author,color='white'))
#     print(colored(Stars,color='green'))
    
    
    
# =============================================================================
# creating unique list
# =============================================================================
def Get_unique_list(Numbers):
    Unique_number_list = []
    Numbers = str(Numbers)
    Unique_numbers = set(Numbers) #converting list into set
    for Number in Unique_numbers:
        Unique_number_list.append(Number)
    Unique_number_list.sort()    #sorting in ascending order
    return Unique_number_list



# =============================================================================
# creating a diagonal matrix
# =============================================================================
def Get_diagonal_mat(Dim,Diag_values):
    Diagonal_mat = np.eye(Dim)
    Entries = np.diag_indices_from(Diagonal_mat)
    Diagonal_mat[Entries] = Diag_values
    return Diagonal_mat 