# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:50:58 2022

@author: sajjaduddin.mahmud
"""
from termcolor import colored

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
def get_unique_list(numbers):
    unique_number_list = []
    numbers = str(numbers)
    unique_numbers = set(numbers)
    for number in unique_numbers:
        unique_number_list.append(number)
    unique_number_list.sort()    
    return unique_number_list