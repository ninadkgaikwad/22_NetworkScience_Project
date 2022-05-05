# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:40:21 2022

@author: ninad gaikwad and sajjad u. mahmud
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import numpy as np
from scipy import linalg

# Custom Modules
import Helper_Functions as HF


# =============================================================================
# Computing Linear System Matrix and its Eigen Components
# =============================================================================
def Compute_linearsystem_matrix(N,l,z,k,m,B,Lp,Ld):
    
    # creating K matrix
    Dim_K = N
    Diag_values_k = k
    K = HF.Get_diagonal_mat(Dim_K, Diag_values_k)
        
    
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
    Z = HF.Get_diagonal_mat(Dim_Z, Diag_values_Z)
        
    
    # Creating M inverse matrix
    Dim_M = N-l
    Diag_values_M = m
    M = HF.Get_diagonal_mat(Dim_M, Diag_values_M)
    M_inv = np.block([np.linalg.inv(M)])
        
    
    # Creating close-loop matrix
    X = Lp + np.matmul(K,Ld) + B
    A11 = - np.matmul(I_l_N,X)
    A12 = I_l_Nl
    A21 = - np.matmul(M_inv,np.matmul(I_Nl_N,X))
    A22 = - Z
    
    A = np.block([[A11,A12],[A21,A22]])
        
    
    # computing eigenvalues and eigevectors of close-loop matrix
    Eig_val_A, Eig_LVec_A, Eig_RVec_A = linalg.eig(A,left=True,right=True)

    return A, Eig_val_A, Eig_LVec_A, Eig_RVec_A


# =============================================================================
# Computing Linear System Matrix and its Eigen Components with Failure
# =============================================================================
def Compute_linearsystem_matrix_withFailure(N,l,p,z,k,m,Adj_Grid,Adj_Comm,MasterNode,NodeFailure_Type,FailureNode):
    
    # creating K matrix
    Dim_K = N
    Diag_values_k = k
    K = HF.Get_diagonal_mat(Dim_K, Diag_values_k)
        
    
    # Creating I_l_N matrix
    I_l_l = np.eye(l)
    I_l_N = np.block([[I_l_l,np.zeros((l,N-l))],[np.zeros((N-l,l)),np.zeros((N-l,N-l))]])
        

    # Creating I_l_Nl matrix
    I_l_Nl = np.block([[np.zeros((l,N-l))],[np.eye(N-l)]])
        
 
    # Creating I_Nl_N matrix
    I_Nl_N = np.block([np.zeros((N-l,l)),np.eye(N-l)])
        
        
    # creating Z matrix
    Dim_Z = N-l
    Diag_values_Z = z
    Z = HF.Get_diagonal_mat(Dim_Z, Diag_values_Z)
        
    
    # creating M inverse matrix
    Dim_M = N-l
    Diag_values_M = m
    M = HF.Get_diagonal_mat(Dim_M, Diag_values_M)
    M_inv = np.block([np.linalg.inv(M)])
        
    
    # Creating B Matrix
    B = np.zeros((l+p,l+p))
    
    B[MasterNode, MasterNode] = 1
    
    Adj_Grid_New = np.copy(Adj_Grid)
    Adj_Comm_New = np.copy(Adj_Comm)
    
    # Modifying Laplacian Matrix
    if NodeFailure_Type == 1:
        
        # Modifying B
        if FailureNode == MasterNode:
            B[MasterNode, MasterNode] = 0 
        
        # Modifying Grid Laplacian
        Adj_Grid_New[:,FailureNode] = 0
        Adj_Grid_New[FailureNode,:] = 0
        
        # Creating Grid Degree Matrix for New Adjacency Matrix
        Degree_Grid_New = np.sum(Adj_Grid_New , axis=1)
        Degree_Diag_Grid_New = np.eye(l+p)
        Entries = np.diag_indices_from(Degree_Diag_Grid_New)
        Degree_Diag_Grid_New[Entries] = Degree_Grid_New
        
        # Creating New Grid Laplacian Matrix
        Laplacian_Grid_New = Degree_Diag_Grid_New - Adj_Grid_New
        
        # Modifying Comm Laplacian
        Adj_Comm_New[:,FailureNode] = 0
        Adj_Comm_New[FailureNode,:] = 0
        
        # Creating Comm Degree Matrix for New Adjacency Matrix
        Degree_Comm_New = np.sum(Adj_Comm_New , axis=1)
        Degree_Diag_Comm_New = np.eye(l+p)
        Entries = np.diag_indices_from(Degree_Diag_Comm_New)
        Degree_Diag_Comm_New[Entries] = Degree_Comm_New
        
        # Creating New Comm Laplacian Matrix
        Laplacian_Comm_New = Degree_Diag_Comm_New - Adj_Comm_New
        
    elif NodeFailure_Type == 2:
        
        # Creating Grid Degree Matrix for New Adjacency Matrix
        Degree_Grid_New = np.sum(Adj_Grid_New , axis=1)
        Degree_Diag_Grid_New = np.eye(l+p)
        Entries = np.diag_indices_from(Degree_Diag_Grid_New)
        Degree_Diag_Grid_New[Entries] = Degree_Grid_New
        
        # Creating New Grid Laplacian Matrix
        Laplacian_Grid_New = Degree_Diag_Grid_New - Adj_Grid_New        
        
        # Modifying Comm Laplacian
        Adj_Comm_New[:,FailureNode] = 0
        Adj_Comm_New[FailureNode,:] = 0
        
        # Creating Comm Degree Matrix for New Adjacency Matrix
        Degree_Comm_New = np.sum(Adj_Comm_New , axis=1)
        Degree_Diag_Comm_New = np.eye(l+p)
        Entries = np.diag_indices_from(Degree_Diag_Comm_New)
        Degree_Diag_Comm_New[Entries] = Degree_Comm_New
        
        # Creating New Comm Laplacian Matrix
        Laplacian_Comm_New = Degree_Diag_Comm_New - Adj_Comm_New
        
    
    
    # creating close-loop matrix
    Lp = Laplacian_Grid_New
    Ld = Laplacian_Comm_New
    
    X = Lp + np.matmul(K,Ld) + B
    A11 = - np.matmul(I_l_N,X)
    A12 = I_l_Nl
    A21 = - np.matmul(M_inv,np.matmul(I_Nl_N,X))
    A22 = - Z
    
    A = np.block([[A11,A12],[A21,A22]])
        
    
    # computing eigenvalues and eigevectors of close-loop matrix
    Eig_val_A, Eig_LVec_A, Eig_RVec_A = linalg.eig(A,left=True,right=True)

    return A, Eig_val_A, Eig_LVec_A, Eig_RVec_A


# =============================================================================
# Computing Linear System Matrix and its Eigen Components
# =============================================================================
def Get_DominantEigenComponents(Eig_val, Eig_LVec, Eig_RVec):
    
    # Get Dominant Eigen Value
    Dominant_EigenValue_Index = np.argmax(np.absolute(Eig_val.real))
    
    Dominant_EigenValue = Eig_val[Dominant_EigenValue_Index]
    
    # Get Left/Right Eigen Vectors Corresponding to Dominant Eigen Value
    Dominant_Eig_Lvec = Eig_LVec[:,Dominant_EigenValue_Index]
    
    Dominant_Eig_Rvec = Eig_RVec[:,Dominant_EigenValue_Index]
    
    # Returning Dominant Eigen Value Components
    return Dominant_EigenValue, Dominant_Eig_Lvec, Dominant_Eig_Rvec       


# =============================================================================
# Function: Compute Master Node
# =============================================================================
def Compute_MasterNode_ActualLinearSystem (N,l,p,z,k,m,Lp,Ld,MethodType1, MethodType2):
    
    # IF ELIF LOOP: For Type of Method used to compute Master Node
    if (MethodType1==1): # Method from Paper (2)
        
        # Create B Matrix as an identity Matrix (All nodes are master Nodes)
        if (MethodType2==1): # For initialization with Identity Matrix
            
            B = np.eye(N)
        
        elif (MethodType2==2): # For initialization with Zeros Matrix
        
            B = np.zeros((N,N))
        
        # Create B Matrix as a Zeros Matrix (None is a Master Node)
        # B = np.zeros((N,N))
        
        # Creating Linear System Matrix and computing Eigen Values and left/right Eigen Vectors
        A_Ini, Eig_val_Ini, Eig_LVec_Ini, Eig_RVec_Ini = Compute_linearsystem_matrix(N,l,z,k,m,B,Lp,Ld)
        
        # Calling custom function to get Dominant Eigen Components
        Dominant_EigenValue_Ini, Dominant_Eig_Lvec_Ini, Dominant_Eig_Rvec_Ini  = Get_DominantEigenComponents(Eig_val_Ini, Eig_LVec_Ini, Eig_RVec_Ini)
        
        # Computing Real Parts of Dominant_EigenValue_Ini, Dominant_Eig_Lvec_Ini, Dominant_Eig_Rvec_Ini
        Real_Dominant_EigenValue_Ini = Dominant_EigenValue_Ini.real
        
        Real_Dominant_Eig_Lvec_Ini = Dominant_Eig_Lvec_Ini.real
        
        Real_Dominant_Eig_Rvec_Ini = Dominant_Eig_Rvec_Ini.real
        
        # Initializing Eta Vector
        Eta = np.zeros((N,))
        
        # FOR LOOP: over computing Eta values for ranking nodes
        for i in range(N):
            
            # IF ELSE LOOP: For Inverter/Synchronous Machine based DGs
            if (i<=l): # For each Node with Inverter Based DGs
            
                Eta[i] = Real_Dominant_Eig_Lvec_Ini[i]*Real_Dominant_Eig_Rvec_Ini[i]
                
            else: # For each Node with Synchronous Machine Based DGs
            
                Eta[i] = Real_Dominant_Eig_Lvec_Ini[i]*Real_Dominant_Eig_Rvec_Ini[i-l]
        
        # Getting Max/Min Indices of Eta
        Max_Index_Eta = np.argmax(Eta)
        
        Min_Index_Eta = np.argmin(Eta)
        
        # Creating List of  Max/Min Indices of Eta
        MaxMin_Index_Eta_List = [Max_Index_Eta, Min_Index_Eta]
        
        # Initializing Vector to hold Dominant Eigen Value Deviation
        Dominant_EigenValue_Deviation_Vector = np.zeros((len(MaxMin_Index_Eta_List),))
        
        # FOR LOOP: over each element of MaxMin_Index_Eta_List
        for i in range(len(MaxMin_Index_Eta_List)):
            
            # Initializing B_Eta with B
            B_Eta = np.copy(B)
            
            # Getting current index of Eta 
            Current_Index_Eta = MaxMin_Index_Eta_List[i]
            
            # Creating B Matrix based on current index of Eta - Replacing b_ii=1 to b_ii=0 in B_Eta
            if (MethodType2==1): # For initialization with Identity Matrix
                
                B_Eta[Current_Index_Eta,Current_Index_Eta]=0
            
            elif (MethodType2==2): # For initialization with Zeros Matrix
            
                B_Eta[Current_Index_Eta,Current_Index_Eta]=1 
            
            # Calling custom function to create New Linear System Matrix with B_Eta 
            A_Current, Eig_val_Current, Eig_LVec_Current, Eig_RVec_Current = Compute_linearsystem_matrix(N,l,z,k,m,B_Eta,Lp,Ld)
            
            # Calling custom function to get Dominant Eigen Components of A_Current
            Dominant_EigenValue_Current, Dominant_Eig_Lvec_Current, Dominant_Eig_Rvec_Current  = Get_DominantEigenComponents(Eig_val_Current, Eig_LVec_Current, Eig_RVec_Current)
            
            # Computing Dominant Eigen Value Deviation
            Dominant_EigenValue_Deviation_Vector[i] = (np.absolute(Real_Dominant_EigenValue_Ini-Dominant_EigenValue_Current.real))/(np.absolute(Real_Dominant_EigenValue_Ini)) 
            
        # Finding the Max Deviation Index
        Max_Deviation_Index = np.argmax(Dominant_EigenValue_Deviation_Vector)
        
        # Finding the Master Node (Node causing maximum deviation in Dominant Eigen Value)
        MasterNode = MaxMin_Index_Eta_List[Max_Deviation_Index]
        
        # Creating B Matrix for the Master Node
        B_MasterNode = np.zeros((N,N))
        
        B_MasterNode[MasterNode,MasterNode]=1
        
        # Calling custom function to create New Linear System Matrix with B_MasterNode 
        ActualLinearSystem_Matrix, Eig_val_Current, Eig_LVec_Current, Eig_RVec_Current = Compute_linearsystem_matrix(N,l,z,k,m,B_MasterNode,Lp,Ld)               
        
    elif (MethodType1==2): # Brute Force Method
            
        # Create B Matrix as an identity Matrix (All nodes are master Nodes)
        if (MethodType2==1): # For initialization with Identity Matrix
            
            B = np.eye(N)
        
        elif (MethodType2==2): # For initialization with Zeros Matrix
        
            B = np.zeros((N,N))
        
        # Create B Matrix as a Zeros Matrix (None is a Master Node)
        # B = np.zeros((N,N))
        
        # Creating Linear System Matrix and computing Eigen Values and left/right Eigen Vectors
        A_Ini, Eig_val_Ini, Eig_LVec_Ini, Eig_RVec_Ini = Compute_linearsystem_matrix(N,l,z,k,m,B,Lp,Ld)
        
        # Calling custom function to get Dominant Eigen Components
        Dominant_EigenValue_Ini, Dominant_Eig_Lvec_Ini, Dominant_Eig_Rvec_Ini  = Get_DominantEigenComponents(Eig_val_Ini, Eig_LVec_Ini, Eig_RVec_Ini)
        
        # Computing Real Parts of Dominant_EigenValue_Ini, Dominant_Eig_Lvec_Ini, Dominant_Eig_Rvec_Ini
        Real_Dominant_EigenValue_Ini = Dominant_EigenValue_Ini.real
        
        Real_Dominant_Eig_Lvec_Ini = Dominant_Eig_Lvec_Ini.real
        
        Real_Dominant_Eig_Rvec_Ini = Dominant_Eig_Rvec_Ini.real        
        
        # Initializing Vector to hold Dominant Eigen Value Deviation
        Dominant_EigenValue_Deviation_Vector = np.zeros(N,)
        
        # FOR LOOP: over each Node
        for i in range(N):
            
            # Initializing B_Eta with B
            B_Eta = np.copy(B)
            
            # Getting current index of Eta 
            Current_Index_Eta = i
            
            # Creating B Matrix based on current index of Eta - Replacing b_ii=1 to b_ii=0 in B_Eta
            if (MethodType2==1): # For initialization with Identity Matrix
                
                B_Eta[Current_Index_Eta,Current_Index_Eta]=0
            
            elif (MethodType2==2): # For initialization with Zeros Matrix
            
                B_Eta[Current_Index_Eta,Current_Index_Eta]=1 
            
            # Calling custom function to create New Linear System Matrix with B_Eta 
            A_Current, Eig_val_Current, Eig_LVec_Current, Eig_RVec_Current = Compute_linearsystem_matrix(N,l,z,k,m,B_Eta,Lp,Ld)
            
            # Calling custom function to get Dominant Eigen Components of A_Current
            Dominant_EigenValue_Current, Dominant_Eig_Lvec_Current, Dominant_Eig_Rvec_Current  = Get_DominantEigenComponents(Eig_val_Current, Eig_LVec_Current, Eig_RVec_Current)
            
            # Computing Dominant Eigen Value Deviation
            Dominant_EigenValue_Deviation_Vector[i] = (np.absolute(Real_Dominant_EigenValue_Ini-Dominant_EigenValue_Current.real))/(np.absolute(Real_Dominant_EigenValue_Ini)) 
            
        # Finding the Max Deviation Index
        Max_Deviation_Index = np.argmax(Dominant_EigenValue_Deviation_Vector)
        
        # Finding the Master Node (Node causing maximum deviation in Dominant Eigen Value)
        MasterNode = Max_Deviation_Index
        
        # Creating B Matrix for the Master Node
        B_MasterNode = np.zeros((N,N))
        
        B_MasterNode[MasterNode,MasterNode]=1
        
        # Calling custom function to create New Linear System Matrix with B_MasterNode 
        ActualLinearSystem_Matrix, Eig_val_Current, Eig_LVec_Current, Eig_RVec_Current = Compute_linearsystem_matrix(N,l,z,k,m,B_MasterNode,Lp,Ld)          
            
    # Return Master Node and Actual Linear System Matrix
    return MasterNode, ActualLinearSystem_Matrix, Eig_val_Current, Eig_LVec_Current, Eig_RVec_Current


# =============================================================================
# Function: Compute Controller Command
# =============================================================================
def Compute_Controller_Command(x,Adj_d,k,MasterNode, l, p, FailureNode, NodeFailure_Type):
    
    # Getting Zeta Vector
    Zeta_Vector = x[range(l+p+p)]
    
    # Initializing Control Command Vector
    u_Vector = np.zeros((l+p+p,))
    
    # IF ELIF LOOP: For Node Failure Type
    if (NodeFailure_Type==0): # No Failure Case
    
        # FOR LOOP: over each state node for Computing Current Control Command
        for i in range(l+p+p):
            
            # Getting Current Node
            CurrentNode = i
            
            # IF ELSE LOOP: For nodes requiring Control Command
            if ((CurrentNode<=(l-1)) or (CurrentNode>(l-1+p))): # DG is Inverter/Synchronous Machine Based
            
                # IF ELSE LOOP: For nodes requiring Control Command Segregated
                if (CurrentNode<=(l-1)):
                    
                    # FOR LOOP: Over each Communication Network Node Connection
                    for j in range(l+p):
                        
                        # Computing Control Command
                        u_Vector[i] = u_Vector[i] - (k*(Adj_d[i,j])*(Zeta_Vector[i]-Zeta_Vector[j]))
                        
                    # IF LOOP: For correcting Control Command for Master Node
                    if (CurrentNode == MasterNode):
                    
                            # Correcting Control Command for Master Node
                            u_Vector[i] = u_Vector[i] + Zeta_Vector[i]
                            
                elif (CurrentNode>(l-1+p)):
                    
                    # Correcting index i
                    ii = i-p
                    
                    # FOR LOOP: Over each Communication Network Node Connection
                    for j in range(l+p):
                        
                        # Computing Control Command
                        u_Vector[i] = u_Vector[i] - (k*(Adj_d[ii,j])*(Zeta_Vector[ii]-Zeta_Vector[j]))
                        
                    # IF LOOP: For correcting Control Command for Master Node
                    if (ii == MasterNode):
                    
                            # Correcting Control Command for Master Node
                            u_Vector[i] = u_Vector[i] + Zeta_Vector[ii]                    
                    
                
            else : # These are extra states coming from second order Synchronous Machine based DG Nodes
                
                # Computing Control Command
                u_Vector[i] = 0      
    
    elif (NodeFailure_Type==1): # Grid Node Failure Case
    
        # FOR LOOP: over each state node for Computing Current Control Command
        for i in range(l+p+p):
            
            # Getting Current Node
            CurrentNode = i
            
            # IF ELSE LOOP: For Current Node to be Grid Failure Node
            if ((CurrentNode == FailureNode) or ((CurrentNode-p) == FailureNode)):

                # Computing Control Command for Master Node
                u_Vector[i] = 0                      
                    
        
            else:

                # IF ELSE LOOP: For nodes requiring Control Command
                if ((CurrentNode<=(l-1)) or (CurrentNode>(l-1+p))): # DG is Inverter/Synchronous Machine Based
                
                    # IF ELSE LOOP: For nodes requiring Control Command Segregated
                    if (CurrentNode<=(l-1)):
                        
                        # FOR LOOP: Over each Communication Network Node Connection
                        for j in range(l+p):
                            
                            # IF LOOP: For continuing over Failure Node
                            if (j == FailureNode):
                                
                                continue
                            
                            # Computing Control Command
                            u_Vector[i] = u_Vector[i] - (k*(Adj_d[i,j])*(Zeta_Vector[i]-Zeta_Vector[j]))
                            
                        # IF LOOP: For correcting Control Command for Master Node
                        if (CurrentNode == MasterNode):
                        
                                # Correcting Control Command for Master Node
                                u_Vector[i] = u_Vector[i] + Zeta_Vector[i]
                                
                    elif (CurrentNode>(l-1+p)):
                        
                        # Correcting index i
                        ii = i-p
                        
                        # FOR LOOP: Over each Communication Network Node Connection
                        for j in range(l+p):
                            
                            # IF LOOP: For continuing over Failure Node
                            if (j == FailureNode):
                                
                                continue
                                                        
                            # Computing Control Command
                            u_Vector[i] = u_Vector[i] - (k*(Adj_d[ii,j])*(Zeta_Vector[ii]-Zeta_Vector[j]))
                            
                        # IF LOOP: For correcting Control Command for Master Node
                        if (ii == MasterNode):
                        
                                # Correcting Control Command for Master Node
                                u_Vector[i] = u_Vector[i] + Zeta_Vector[ii]                    
                    
                
                else : # These are extra states coming from second order Synchronous Machine based DG Nodes
                    
                    # Computing Control Command
                    u_Vector[i] = 0        
    
    elif (NodeFailure_Type==2): # Comm Node Failure Case
    
        # FOR LOOP: over each state node for Computing Current Control Command
        for i in range(l+p+p):
            
            # Getting Current Node
            CurrentNode = i
            
            # IF ELSE LOOP: For Current Node to be Communication Failure Node
            if ((CurrentNode == FailureNode) or ((CurrentNode-p) == FailureNode)):
            
                # IF ELSE LOOP: For Current Node to be the Master Node
                if (CurrentNode == MasterNode):
                    
                    # Computing Control Command for Master Node
                    u_Vector[i] = Zeta_Vector[i]  
                    
                else:
                    # Computing Control Command for Master Node
                    u_Vector[i] = 0                      
                    
        
            else:

                # IF ELSE LOOP: For nodes requiring Control Command
                if ((CurrentNode<=(l-1)) or (CurrentNode>(l-1+p))): # DG is Inverter/Synchronous Machine Based
                
                    # IF ELSE LOOP: For nodes requiring Control Command Segregated
                    if (CurrentNode<=(l-1)):
                        
                        # FOR LOOP: Over each Communication Network Node Connection
                        for j in range(l+p):
                            
                            # IF LOOP: For continuing over Failure Node
                            if (j == FailureNode):
                                
                                continue
                            
                            # Computing Control Command
                            u_Vector[i] = u_Vector[i] - (k*(Adj_d[i,j])*(Zeta_Vector[i]-Zeta_Vector[j]))
                            
                        # IF LOOP: For correcting Control Command for Master Node
                        if (CurrentNode == MasterNode):
                        
                                # Correcting Control Command for Master Node
                                u_Vector[i] = u_Vector[i] + Zeta_Vector[i]
                                
                    elif (CurrentNode>(l-1+p)):
                        
                        # Correcting index i
                        ii = i-p
                        
                        # FOR LOOP: Over each Communication Network Node Connection
                        for j in range(l+p):
                            
                            # IF LOOP: For continuing over Failure Node
                            if (j == FailureNode):
                                
                                continue
                                                        
                            # Computing Control Command
                            u_Vector[i] = u_Vector[i] - (k*(Adj_d[ii,j])*(Zeta_Vector[ii]-Zeta_Vector[j]))
                            
                        # IF LOOP: For correcting Control Command for Master Node
                        if (ii == MasterNode):
                        
                                # Correcting Control Command for Master Node
                                u_Vector[i] = u_Vector[i] + Zeta_Vector[ii]                    
                    
                
                else : # These are extra states coming from second order Synchronous Machine based DG Nodes
                    
                    # Computing Control Command
                    u_Vector[i] = 0                 
    
            
    # Return Control Command Vector
    return u_Vector
        
# =============================================================================
# Function: Compute Controller Frequency Rersponse performance
# =============================================================================
def Compute_Controller_FrequencyResponse_Performance( Timevector, Del_t, Delta_mat, NodeFailure_Type, FailureNode):
   
   Stability_Status = 1 
   
   # IF LOOP: Checking Stability Status 
   if Stability_Status == 0:
        
       Delta_dot_mat = np.NaN
        
       Delta_dot_avg = np.NaN
        
       Delta_dot_avg_time = np.NaN
        
   elif Stability_Status == 1:
        
        if NodeFailure_Type == 0 or NodeFailure_Type == 2:
            
            # Computing delta_dot matrix from delta matrix
             row , column = Delta_mat.shape
             
             Delta_dot_mat_row = row - 1
             
             Delta_dot_mat = np.zeros((Delta_dot_mat_row,column))
             
             for i in range(column):
                 
                 Delta_i = Delta_mat[:,i]
                 
                 for j in range(row):
                     
                     if (j+1) < row:
                         
                        Slope_j = (Delta_i[j+1] -Delta_i[j])/Del_t
                        Delta_dot_mat[j,i] = Slope_j
                         
             Delta_dot_avg = np.sum(Delta_dot_mat,axis=1)/column
               
             Delta_dot_avg_absolute = np.absolute(Delta_dot_avg)
               
             Delta_dot_avg_absolute_mean = np.mean(Delta_dot_avg_absolute)
                
             # Finding the Intersection Point from End Side
                    
             # FOR LOOP: To find the Time Value for the Intersection Point
             ll = len(Delta_dot_avg_absolute)
             
             for i in range(ll):
                 
                 ii = ll - i - 1
                 
                 if Delta_dot_avg_absolute[ii] > Delta_dot_avg_absolute_mean:
                     
                     Timevector_index = ii
                     
                     break
                 
             Delta_dot_avg_time = Timevector[Timevector_index]                                     
         
        elif NodeFailure_Type == 1:            
             
            # Computing delta_dot matrix from delta matrix
             Delta_mat = np.delete(Delta_mat, FailureNode, 1)
            
             row , column = Delta_mat.shape
             
             Delta_dot_mat_row = row - 1
             
             Delta_dot_mat = np.zeros((Delta_dot_mat_row,column))
             
             for i in range(column):
                 
                 Delta_i = Delta_mat[:,i]
                 
                 for j in range(row):
                     
                     if (j+1) < row:
                         
                         Slope_j = (Delta_i[j+1] -Delta_i[j])/Del_t
                         Delta_dot_mat[j,i] = Slope_j
                         
             Delta_dot_avg = np.sum(Delta_dot_mat,axis=1)/column
             
             Delta_dot_avg_absolute = np.absolute(Delta_dot_avg)
             
             Delta_dot_avg_absolute_mean = np.mean(Delta_dot_avg_absolute)
             
             # Finding the Intersection Point from End Side
             
             # FOR LOOP: To find the Time Value for the Intersection Point
             ll = len(Delta_dot_avg_absolute)
             
             for i in range(ll):
                 
                 ii = ll - i - 1
                 
                 if Delta_dot_avg_absolute[ii] > Delta_dot_avg_absolute_mean:
                     
                     Timevector_index = ii
                     
                     break
                 
             Delta_dot_avg_time = Timevector[Timevector_index]


            
   return Delta_dot_mat, Delta_dot_avg, Delta_dot_avg_time

 
     
    