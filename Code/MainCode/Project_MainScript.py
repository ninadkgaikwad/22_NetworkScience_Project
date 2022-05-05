# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 21:11:17 2022

@author: ninad gaikwad and sajjad u. mahmud
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import scipy.io as sio
import numpy as np
import time
import test as t

# Custom Modules
import DataPreparation_Functions as DPF
import MicrogridController_Functions as MCF
import MicrogridPlant_Functions as MPF
import MicrogridRanking_Functions as MRF

# =============================================================================
# Simulation Setup
# =============================================================================

# User Information [Ninad or Sajjad or LabLaptop]
User = "Sajjad"

# Percentage of Synchronous DGs and Inverter based DGs
SyncDG_Percentage = 25

# Physical Constants
MachineInertiaRange_M = (0.1,0.25)

MachineInertaDamperConstant_z = 2.5

NodePowerTransfer_T = 1

MachineInverterCoefficient_G = 1

# Initial States
MachineInitialStateRange_x_Ini = (-0.0175,0.0175)

# Controller Constants
ControllerGain_k = 10

MasterNodeGain_b = 1.

# Grid/Communication Network Constants
GridCommNetworkFailure_Percentage_Vector=[0.,10.,20.,30.,40.,50.,60.,70.,80.,90.,100.]
# GridCommNetworkFailure_Percentage_Vector=[0.,30.,80.]

# Time Constants
TimeStart = 0.

TimeEnd = 60.

TimeDelta = 0.01

# Randomness Seeds
RandomSeed = 1

# Algorithm for Choosing Master Control Node 
MasterControlNode_Algorithm = 1 # [1 - From Paper (2) , 2 - Brute Force Version of Paper (2)]

MasterControlNode_Initialization = 1 # [1 - Initialize B matrix with Identity Matrix , 2 - Initialize B matrix with Zeros Matrix]

# Choosing Microgrid Plant Dynamics Type
PlantDynamics_Type = 1 # [1 - Linear , 2 - Nonlinear]

# Choosing ODE Solver Type 
ODE_Solver_Type = 1 # [1 - Python In-Built , 2 - Custom Made]

# Performance Computation Constants
Percentage_limit = 10 

Threshold = 10e-10

# Folder Paths to Grid and Communication Graphs Data

# Grid/Communication Network Names List
Grid_Name = "IEEE_342"

Comm_Name= ['P_1','P_2','P_3','P_4','P_5','P_6','P_7','P_8','P_9','P_10','P_20','P_30','P_40','P_50','P_60','P_70','P_80','P_90','P_100']
# Comm_Name= ['P_10','P_40','P_90'] # Debugger
Comm_List = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# Comm_List = [0,1,2]
# Comm_List1 = [9,12,18] # Debugger


# Ranking Methods Name
Rank_Method_Name = ['Degree','PageRank','Eigenvalue_Analysis']

GridCommData_FolderName = r"\IEEE_342" # IEEE_13 , IEEE_123, IEEE_342, IEEE_8500

if (User == "Ninad"):
    
    GridCommData_Path = r"C:\Users\ninad\OneDrive - Washington State University (email.wsu.edu)\TeamProjects\CPT_S_591_Spring2022\22_NetworkScience_Project\22_NetworkScience_Project\Preprocessed_Data"

    # Results Path
    # Result_Path = 
    
elif (User == "Sajjad"):
    
    GridCommData_Path = r"C:\Users\sajjaduddin.mahmud\OneDrive - Washington State University (email.wsu.edu)\CPT_S_591_Spring2022\22_NetworkScience_Project\22_NetworkScience_Project\Preprocessed_Data"

    # Results Path
    Result_Path = r"C:\Users\sajjaduddin.mahmud\ieee342"
    
elif (User == "LabLaptop"):

    GridCommData_Path = r"C:\Users\Auser\Dropbox\My PC (EME23-L01)\Desktop\NetworkScience\CPT_S_591_Spring2022\22_NetworkScience_Project\22_NetworkScience_Project\Preprocessed_Data"

    # Results Path
    Result_Path = r"C:\Users\Auser\Dropbox\My PC (EME23-L01)\Desktop\NetworkScience\CPT_S_591_Spring2022\22_NetworkScience_Project\22_NetworkScience_Project\Result_Data" + GridCommData_FolderName

# =============================================================================
# Basic Computation
# ============================================================================= 

# Compute Inverter DG Percentage
InverterDG_Percentage = 100-SyncDG_Percentage

# Computing Time Vector
Time_Vector = np.arange(TimeStart,TimeEnd,TimeDelta)

# =============================================================================
# Getting Grid and Communication Graphs
# =============================================================================

# Calling custom function to get Grid and Communication Graphs
GridNetworkData, CommNetworkData_List = DPF.Get_GridCommGraph_Data(GridCommData_Path, GridCommData_FolderName)


# Getting Current Grid Network Data
Adj_p = GridNetworkData['Adj_DG']

Degree_p = GridNetworkData['degree_DG']

Lp = GridNetworkData['laplacian_DG']

# Initializing Storage Lists
Storage_Grid_Rank_array_list = []

Storage_Comm_Rank_array_list = []

Storage_FrequencyResponse_BaseCase = []

Storage_RankingMethod_ComputationalTime_list = []

Storage_RankWise_CommFailure_Nodes_List = []

# RankWise_CommFailure_Nodes_Vector, Grid_Rank_array_List, CommNetworkData_List
# Storage_FrequencyResponse_GridNodeFailureCase = [[ [0 for col in range(len(GridCommNetworkFailure_Percentage_Vector))] for col in range(3)] for row in range(len(CommNetworkData_List))]

# Storage_FrequencyResponse_CommNodeFailureCase = [[ [0 for col in range(len(GridCommNetworkFailure_Percentage_Vector))] for col in range(3)] for row in range(len(CommNetworkData_List))]

Storage_FrequencyResponse_GridNodeFailureCase = [[ [0 for col in range(len(GridCommNetworkFailure_Percentage_Vector))] for col in range(3)] for row in range(len(Comm_List))]

Storage_FrequencyResponse_CommNodeFailureCase = [[ [0 for col in range(len(GridCommNetworkFailure_Percentage_Vector))] for col in range(3)] for row in range(len(Comm_List))]

Storage_RankWise_CommFailure_Nodes_List = [ [0 for col in range(len(GridCommNetworkFailure_Percentage_Vector))] for col in range(3)]


# =============================================================================
# Compute Ranking for Grid Nodes
# =============================================================================

# Degree Centrality

Grid_Degree_Rank_array = MRF.Get_ranking_degree(Adj_p)
    

# Page Rank

Grid_PageRank_Rank_array = MRF.Get_ranking_pagerank(Adj_p)
    

# Eigen Value Analysis

Grid_EigenAnalysis_Rank_array = MRF.Get_ranking_eigenanalysis(Adj_p,Lp)


# Creating List for Rank Arrays
Grid_Rank_array_List = [Grid_Degree_Rank_array,Grid_PageRank_Rank_array,Grid_EigenAnalysis_Rank_array]  

# =============================================================================
# Saving Ranking for Grid / Communication Nodes
# =============================================================================
Storage_Grid_Rank_array_list = [Grid_Degree_Rank_array,Grid_PageRank_Rank_array,Grid_EigenAnalysis_Rank_array]


# =============================================================================
# FOR LOOP: over all Communication Networks
# =============================================================================
for i in Comm_List:    # len(CommNetworkData_List)
    
    # =============================================================================
    # Basic Computation
    # =============================================================================    

    # Getting Current Communication Network Data
    CommNetwork_Dict = CommNetworkData_List[Comm_List[i]]  
    
    Adj_d = CommNetwork_Dict['Adj_Comm'] 
    
    Degree_d = CommNetwork_Dict['degree_Comm']  
        
    Ld =  CommNetwork_Dict['laplacian_Comm'] 
    
    # Computing N (Total # of Nodes) and l (Total # of Inv DGs) and p (Total # of Sync DGs) and n (Total # of States)
    N = np.shape(Lp)[0]
    
    p = round(N*(SyncDG_Percentage/100))
    
    n = N+p
    
    l = N-p
    
    # Creating Random Machine Inertia Vector
    np.random.seed(RandomSeed) ; M_Vector = np.random.uniform(MachineInertiaRange_M[0],MachineInertiaRange_M[1],(p,))  
    
    # Creating Random Initial State Vector
    np.random.seed(RandomSeed) ; InitialState_Vector = np.random.uniform(MachineInitialStateRange_x_Ini[0],MachineInitialStateRange_x_Ini[1],(N+p,))

    # =============================================================================
    # Compute Master Node and the Acutal Linear System of the Plant
    # =============================================================================
    
    # Calling custom function to compute Master Node
    MasterNode, ActualLinearSystem_Matrix, Eig_val_Base, Eig_LVec_Base, Eig_RVec_Base = MCF.Compute_MasterNode_ActualLinearSystem (N,l,p,MachineInertaDamperConstant_z,ControllerGain_k,M_Vector,Lp,Ld,MasterControlNode_Algorithm,MasterControlNode_Initialization)    

    # Calling custom function to get Dominant Eigen Components
    Dominant_EigenValue_Base  = MCF.Get_DominantEigenComponents(Eig_val_Base, Eig_LVec_Base, Eig_RVec_Base)
    
    # Computing Real Parts of Dominant_EigenValue
    Real_Dominant_EigenValue_Base = Dominant_EigenValue_Base[0].real
    
    # Compute Stability of the Current Microgrid Network
    Stability_Status_Base = MPF.Microgrid_Plant_Stability(Eig_val_Base)      
    
    # =============================================================================
    # Compute Ranking for Communication Nodes
    # =============================================================================
    
    # Degree Centrality
        
    start = time.time()
    
    Comm_Degree_Rank_array = MRF.Get_ranking_degree(Adj_d)  
    
    end = time.time()
    
    ComputationalTime_Degree = (end-start)

    # Page Rank
        
    start = time.time()
    
    Comm_PageRank_Rank_array = MRF.Get_ranking_pagerank(Adj_d)    
    
    end = time.time()
    
    ComputationalTime_PageRank = (end-start)
    
    # Eigen Value Analysis
    
    start = time.time()
    
    Comm_EigenAnalysis_Rank_array = MRF.Get_ranking_eigenanalysis(Adj_d,Ld) 
    
    end = time.time()
    
    ComputationalTime_EigenAnalysis = (end-start)

    # Creating List for Rank Arrays
    
    Comm_Rank_array_List = [Comm_Degree_Rank_array,Comm_PageRank_Rank_array,Comm_EigenAnalysis_Rank_array]   
    
    ComputationalTime_Rank_array_List = [ComputationalTime_Degree,ComputationalTime_PageRank,ComputationalTime_EigenAnalysis]


    # =============================================================================
    # Saving Ranking for Grid / Communication Nodes
    # =============================================================================
    
    Storage_Comm_Rank_array_list.append([Comm_Degree_Rank_array,Comm_PageRank_Rank_array,Comm_EigenAnalysis_Rank_array]) 

    Storage_RankingMethod_ComputationalTime_list.append(ComputationalTime_Rank_array_List)


    # =============================================================================
    # Microgrid Time-Simulation - No Failure Case       
    # =============================================================================
    
    # Creating NodeFailure_Type (0 - No Failure , 1 - Grid Node Failure , 2 - Comm Node Failure)
    NodeFailure_Type = 0
    
    FailureNode = -1

    AngleResponse_Grid_Matrix = MPF.Microgrid_Plant_TimeSimulation(ODE_Solver_Type,ActualLinearSystem_Matrix,InitialState_Vector,M_Vector,MachineInertaDamperConstant_z,NodePowerTransfer_T,MachineInverterCoefficient_G,ControllerGain_k,Adj_p,Adj_d,FailureNode,Time_Vector,TimeDelta,NodeFailure_Type,N,p,l,MasterNode,PlantDynamics_Type)

           
    # =============================================================================
    # Compute Frerquency Response  Performance
    # =============================================================================
    Grid_Delta_dot_mat, Grid_Delta_dot_avg, Grid_Delta_dot_avg_time = MCF.Compute_Controller_FrequencyResponse_Performance(Time_Vector, TimeDelta, AngleResponse_Grid_Matrix, NodeFailure_Type, FailureNode)


    # =============================================================================
    # Saving Frerquency Response  Performance - No Failure Case
    # =============================================================================
    Storage_FrequencyResponse_BaseCase.append([Grid_Delta_dot_mat, Grid_Delta_dot_avg, Grid_Delta_dot_avg_time, Real_Dominant_EigenValue_Base, Stability_Status_Base])

    # =============================================================================
    # FOR LOOP: over each Ranking method
    # =============================================================================
    for j in range(len(Grid_Rank_array_List)):
        
        # Getting Current Grid/Comm Rank Array
        Grid_Rank_array = Grid_Rank_array_List[j]
        
        Comm_Rank_array = Comm_Rank_array_List[j]
        
        # Computing Rank-Wise Failure Nodes
        RankWise_Failure_Nodes = np.around(np.array(GridCommNetworkFailure_Percentage_Vector)*((N-1)/100))        
        
        # Converting RankWise_Failure_Nodes to int
        RankWise_Failure_Nodes_Vector = RankWise_Failure_Nodes.astype(int)
        
        # Getting Failure Nodes from Grid/Comm Rank_array
        RankWise_GridFailure_Nodes_Vector = Grid_Rank_array[RankWise_Failure_Nodes_Vector,0]
        
        RankWise_GridFailure_Nodes_Vector = RankWise_GridFailure_Nodes_Vector.astype(int)
        
        RankWise_CommFailure_Nodes_Vector = Comm_Rank_array[RankWise_Failure_Nodes_Vector,0]
        
        RankWise_CommFailure_Nodes_Vector = RankWise_CommFailure_Nodes_Vector.astype(int)
    
        RankWise_CommFailure_Nodes_List = [RankWise_GridFailure_Nodes_Vector, RankWise_CommFailure_Nodes_Vector]

        # Storage_RankWise_CommFailure_Nodes_List[i][j] = [RankWise_CommFailure_Nodes_List]

        # =============================================================================
        # FOR LOOP: over each failure node given by Ranking method - Grid Network
        # =============================================================================
        for k in range(len(RankWise_GridFailure_Nodes_Vector)):
            
            # Creating NodeFailure_Type (0 - No Failure , 1 - Grid Node Failure , 2 - Comm Node Failure)
            NodeFailure_Type = 1
            
            # Getting Current Grid Failure Node
            FailureNode = RankWise_GridFailure_Nodes_Vector[k]
            
            
            # =============================================================================
            # Microgrid Failure Linear Matrix Computation        
            # ============================================================================= 
            AFL_System_Grid_Matrix ,Eig_val_AFL_Grid_Matrix , Eig_LVec_AFL_Grid, Eig_RVec_AFL_Grid = MCF.Compute_linearsystem_matrix_withFailure(N,l,p,MachineInertaDamperConstant_z,ControllerGain_k,M_Vector,Adj_p,Adj_d,MasterNode,NodeFailure_Type,FailureNode)
            
            # Calling custom function to get Dominant Eigen Components
            Dominant_EigenValue_AFL_Grid  = MCF.Get_DominantEigenComponents(Eig_val_AFL_Grid_Matrix , Eig_LVec_AFL_Grid, Eig_RVec_AFL_Grid)
            
            # Computing Real Parts of Dominant_EigenValue
            Real_Dominant_EigenValue_AFL_Grid = Dominant_EigenValue_AFL_Grid[0].real
            
            # Compute Stability of the Current Microgrid Network
            Stability_Status_Grid = MPF.Microgrid_Plant_Stability(Eig_val_AFL_Grid_Matrix)            

        
            # =============================================================================
            # Microgrid Time-Simulation        
            # =============================================================================
            AngleResponse_Grid_Matrix = MPF.Microgrid_Plant_TimeSimulation(ODE_Solver_Type,AFL_System_Grid_Matrix,InitialState_Vector,M_Vector,MachineInertaDamperConstant_z,NodePowerTransfer_T,MachineInverterCoefficient_G,ControllerGain_k,Adj_p,Adj_d,FailureNode,Time_Vector,TimeDelta,NodeFailure_Type,N,p,l,MasterNode,PlantDynamics_Type)
            
            
            # =============================================================================
            # Compute Frerquency Response  Performance
            # =============================================================================
            Grid_Delta_dot_mat, Grid_Delta_dot_avg, Grid_Delta_dot_avg_time = MCF.Compute_Controller_FrequencyResponse_Performance(Time_Vector, TimeDelta, AngleResponse_Grid_Matrix, NodeFailure_Type, FailureNode)


            # =============================================================================
            # Saving Frerquency Response  Performance - Grid Node Failure Case
            # =============================================================================
            Storage_FrequencyResponse_GridNodeFailureCase[i][j][k]=[Grid_Delta_dot_mat, Grid_Delta_dot_avg, Grid_Delta_dot_avg_time, Real_Dominant_EigenValue_AFL_Grid, Stability_Status_Grid]

        # =============================================================================
        # FOR LOOP: over each failure node given by Ranking method - Communication Network
        # =============================================================================
        for k in range(len(RankWise_CommFailure_Nodes_Vector)):

            # Creating NodeFailure_Type (0 - No Failure , 1 - Grid Node Failure , 2 - Comm Node Failure)
            NodeFailure_Type = 2
            
            # Getting Current Grid Failure Node
            FailureNode = RankWise_CommFailure_Nodes_Vector[k]
            
            
            # =============================================================================
            # Microgrid Failure Linear Matrix Computation        
            # ============================================================================= 
            AFL_System_Comm_Matrix ,Eig_val_AFL_Comm_Matrix , Eig_LVec_AFL_Comm, Eig_RVec_AFL_Comm = MCF.Compute_linearsystem_matrix_withFailure(N,l,p,MachineInertaDamperConstant_z,ControllerGain_k,M_Vector,Adj_p,Adj_d,MasterNode,NodeFailure_Type,FailureNode)
            
            # Calling custom function to get Dominant Eigen Components
            Dominant_EigenValue_AFL_Comm  = MCF.Get_DominantEigenComponents(Eig_val_AFL_Comm_Matrix , Eig_LVec_AFL_Comm, Eig_RVec_AFL_Comm)
            
            # Computing Real Parts of Dominant_EigenValue
            Real_Dominant_EigenValue_AFL_Comm = Dominant_EigenValue_AFL_Comm[0].real
            
            # Compute Stability of the Current Microgrid Network
            Stability_Status_Comm = MPF.Microgrid_Plant_Stability(Eig_val_AFL_Comm_Matrix)
            
            
            # =============================================================================
            # Microgrid Time-Simulation        
            # =============================================================================
            AngleResponse_Comm_Matrix = MPF.Microgrid_Plant_TimeSimulation(ODE_Solver_Type,AFL_System_Comm_Matrix,InitialState_Vector,M_Vector,MachineInertaDamperConstant_z,NodePowerTransfer_T,MachineInverterCoefficient_G,ControllerGain_k,Adj_p,Adj_d,FailureNode,Time_Vector,TimeDelta,NodeFailure_Type,N,p,l,MasterNode,PlantDynamics_Type)
            
            
            # =============================================================================
            # Compute Frerquency Response  Performance
            # =============================================================================   
            Comm_Delta_dot_mat, Comm_Delta_dot_avg, Comm_Delta_dot_avg_time = MCF.Compute_Controller_FrequencyResponse_Performance(Time_Vector, TimeDelta, AngleResponse_Comm_Matrix, NodeFailure_Type, FailureNode)


            # =============================================================================
            # Saving Frerquency Response  Performance - Comm Node Failure Case
            # =============================================================================
            Storage_FrequencyResponse_CommNodeFailureCase[i][j][k]=[Comm_Delta_dot_mat, Comm_Delta_dot_avg, Comm_Delta_dot_avg_time, Real_Dominant_EigenValue_AFL_Comm, Stability_Status_Comm]


# =============================================================================
#             Create Tables
# =============================================================================            

# Calling Custom Function for Creating Tabulated Results
Data_Frame_Table1, Data_Frame_Table2, Data_Frame_Table3 = DPF.Creating_ResultTable_MicrogridProject(Result_Path, Grid_Name , Comm_Name , Storage_Grid_Rank_array_list , Storage_Comm_Rank_array_list, Storage_RankingMethod_ComputationalTime_list)


# =============================================================================
#             Plot Graphs
# =============================================================================

# Calling Custom Function for Creating Graphs of Results
DPF.Creating_Plots(Result_Path, Grid_Name, Comm_Name, Rank_Method_Name, GridCommNetworkFailure_Percentage_Vector, Time_Vector, Storage_FrequencyResponse_BaseCase , Storage_FrequencyResponse_GridNodeFailureCase , Storage_FrequencyResponse_CommNodeFailureCase)

DPF.Creating_Plots1(Result_Path, Grid_Name, Comm_Name, Rank_Method_Name, GridCommNetworkFailure_Percentage_Vector, Time_Vector, Storage_FrequencyResponse_BaseCase , Storage_FrequencyResponse_GridNodeFailureCase , Storage_FrequencyResponse_CommNodeFailureCase)