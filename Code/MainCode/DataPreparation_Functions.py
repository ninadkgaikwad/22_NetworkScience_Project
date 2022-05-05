# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 12:43:03 2022

@author: ninad gaikwad and sajjad u. mahmud
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import os
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 



# =============================================================================
# Function: Getting Grid and Communication Graphs
# =============================================================================
def Get_GridCommGraph_Data(GridCommData_Path, GridCommData_FolderName):
    
    # Getting full path to the Grid and Communication Graph files
    DataFiles_FullPath = GridCommData_Path+ GridCommData_FolderName  
    
    # Getting all filen names from the data folder
    DataFileName_List = os.listdir(DataFiles_FullPath)
    
    # Initialize CommNetworkData_List
    CommNetworkData_List=[0]*(len(DataFileName_List)-1)
    
    # FOR LOOP: over each data file in the list
    for i in range(len(DataFileName_List)):
        
        # Getting Current File Name
        CurrentFile_Name = DataFileName_List[i]
        
        # Getting Current File Path
        CurrentFile_FullPath=DataFiles_FullPath+'\\'+CurrentFile_Name
        
        # 
        if (CurrentFile_Name.endswith("Comm_P_1")): # Comm_P10
        
            # Loading Communication Network Data
            CommNetworkData_List[0] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_2")): # Comm_P20
        
            # Loading Communication Network Data
            CommNetworkData_List[1] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_3")): # Comm_P30
        
            # Loading Communication Network Data
            CommNetworkData_List[2] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_4")): # Comm_P40
        
            # Loading Communication Network Data
            CommNetworkData_List[3] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_5")): # Comm_P50
        
            # Loading Communication Network Data
            CommNetworkData_List[4] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_6")): # Comm_P60
        
            # Loading Communication Network Data
            CommNetworkData_List[5] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_7")): # Comm_P70
        
            # Loading Communication Network Data
            CommNetworkData_List[6] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_8")): # Comm_P80
        
            # Loading Communication Network Data
            CommNetworkData_List[7] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_9")): # Comm_P90
        
            # Loading Communication Network Data
            CommNetworkData_List[8] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_10")): # Comm_P100
        
            # Loading Communication Network Data
            CommNetworkData_List[9] = sio.loadmat(CurrentFile_FullPath) 

        elif (CurrentFile_Name.endswith("Comm_P_20")): # Comm_P10
        
            # Loading Communication Network Data
            CommNetworkData_List[10] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_30")): # Comm_P20
        
            # Loading Communication Network Data
            CommNetworkData_List[11] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_40")): # Comm_P30
        
            # Loading Communication Network Data
            CommNetworkData_List[12] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_50")): # Comm_P40
        
            # Loading Communication Network Data
            CommNetworkData_List[13] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_60")): # Comm_P50
        
            # Loading Communication Network Data
            CommNetworkData_List[14] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_70")): # Comm_P60
        
            # Loading Communication Network Data
            CommNetworkData_List[15] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_80")): # Comm_P70
        
            # Loading Communication Network Data
            CommNetworkData_List[16] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_90")): # Comm_P80
        
            # Loading Communication Network Data
            CommNetworkData_List[17] = sio.loadmat(CurrentFile_FullPath)
                    
        elif (CurrentFile_Name.endswith("Comm_P_100")): # Comm_P90
        
            # Loading Communication Network Data
            CommNetworkData_List[18] = sio.loadmat(CurrentFile_FullPath)          
            
        elif (CurrentFile_Name.endswith("DG")): # DG
        
            # Loading Grid Network Data
            GridNetworkData_Dict = sio.loadmat(CurrentFile_FullPath)

        
    # Return Grid and Communication Network Data
    return GridNetworkData_Dict, CommNetworkData_List


# =============================================================================
# Computing Similarity between Ranking Methods
# =============================================================================
def Computing_Ranking_Similarity(Rank_Array1, Rank_Array2):
    
    # Initializing Counter
    Counter = 0
    
    # FOR LOOP: Over Each Element of Rank Arrays
    for i in range(len(Rank_Array1)):
       
        # IF LOOP: Checking the Similarity Between Ranked Nodes
        if Rank_Array1[i] == Rank_Array2[i]:
    
            # Incrementing Counter
            Counter += 1
    
    # Computing Similarity Percentage
    Similarity_percentage = Counter / len(Rank_Array1) * 100
    
    return Similarity_percentage
        
    
# =============================================================================
# Table Function
# =============================================================================
def Creating_ResultTable_MicrogridProject(Result_Path, Grid_Name , Comm_Name , Storage_Grid_Rank_array_list , Storage_Comm_Rank_array_list, Storage_RankingMethod_ComputationalTime_list):

    # Creating Excel File name
    File_Name = "\\" + Grid_Name + "_TabulatedResults.xlsx" 

    # Creating Excel Sheet Names
    Sheet1_Name = "GridRankValueTable"    
    
    Sheet2_Name = "CommRankValueTable" 
    
    Sheet3_Name = "RankingComparisonGridCommTable" 
    
    Sheet4_Name = "RankingComputationTimeTable"

    # Combining Grid/Comm Ranking Lists
    Combined_Ranking_List = [Storage_Grid_Rank_array_list, Storage_Comm_Rank_array_list]   
    
    # FOR LOOP:over each ranking list
    for i in range(len(Combined_Ranking_List)):
        
        # IF ELSE LOOP: For segregating Grid and CommRanking Lists
        if (i==0): # We have Grid_Rank_array_list
            
            # Getting Grid Rank Array List
            Grid_Rank_array_list = Combined_Ranking_List[i]
            
            # Getting Components of Grid Rank Array List
            Grid_DegreeRank_array = Grid_Rank_array_list[0]
            
            Grid_PageRank_array = Grid_Rank_array_list[1]
            
            Grid_EigenAnalysisRank_array = Grid_Rank_array_list[2]
            
            # Getting Shape of Grid Rank Array List Components
            r_Grid, c_Grid = Grid_DegreeRank_array.shape
            
            # Creating array of Grid Name
            Grid_Name_array_list = [Grid_Name]*r_Grid
            
            Grid_Name_array = np.reshape(np.array(Grid_Name_array_list),(r_Grid,1))
            
            # Creating Grid Rank Table
            for j in range(len(Grid_Rank_array_list)):
                
                if j == 0:
                    
                    Grid_Rank_Array_Combined_array = Grid_Rank_array_list[j]
                    
                else:
                    
                    Grid_Rank_Array_Combined_array = np.block([Grid_Rank_Array_Combined_array,Grid_Rank_array_list[j]])   
                    
            # Creating Complete Grid Rank Array for Table
            Grid_Rank_Array_Combined_array_ForTable = np.hstack((Grid_Name_array,Grid_Rank_Array_Combined_array))
            
            # Computing Similarity between Grid Nodes Ranking Methods
            Grid_Degree_Pagerank_Simirality = Computing_Ranking_Similarity(Grid_DegreeRank_array[:,0], Grid_PageRank_array[:,0])
            
            Grid_Degree_EigenAnalysis_Simirality = Computing_Ranking_Similarity(Grid_DegreeRank_array[:,0], Grid_EigenAnalysisRank_array[:,0])
            
            Grid_Pagerank_EigenAnalysis_Simirality = Computing_Ranking_Similarity(Grid_PageRank_array[:,0], Grid_EigenAnalysisRank_array[:,0])
            
            # Creating Grid Ranking Similarity Table
            Grid_Rank_Similarity_Table = np.hstack((np.reshape(np.array([Grid_Name]),(1,1)),np.reshape(np.array([Grid_Degree_Pagerank_Simirality,Grid_Degree_EigenAnalysis_Simirality,Grid_Pagerank_EigenAnalysis_Simirality]),(1,3))))
            
            # Creating Combined Grid/Comm Ranking Similarity Table
            GridComm_Combined_Rank_Similarity_Table = Grid_Rank_Similarity_Table
            
        else: # We have Comm_Rank_Storage_list
     
            # Getting Comm_Rank_Sorage_list
            Comm_Rank_Sorage_list = Combined_Ranking_List[i]
            

            
            # FOR LOOP: over each element of Comm_Rank_Sorage_list
            for j in range(len(Comm_Rank_Sorage_list)):

                # Getting current RankingMethod_ComputationalTime_list
                RankingMethod_ComputationalTime_list = Storage_RankingMethod_ComputationalTime_list[j]
                
                RankingMethod_ComputationalTime_array = np.block([np.reshape(np.array(Comm_Name[j]),(1,1)),np.reshape(np.array(RankingMethod_ComputationalTime_list),(1,len(RankingMethod_ComputationalTime_list)))])
                
                # Getting current Comm_Rank_array_list
                Comm_Rank_array_list = Comm_Rank_Sorage_list[j]
                
                # Getting Components of Comm Rank Array List
                Comm_DegreeRank_array = Comm_Rank_array_list[0]
                
                Comm_PageRank_array = Comm_Rank_array_list[1]
                
                Comm_EigenAnalysisRank_array = Comm_Rank_array_list[2]
                
                # Getting Shape of Comm Rank Array List Components
                r_Comm, c_Comm = Comm_DegreeRank_array.shape
                
                # Creating array of Comm Name
                Comm_Name_array_list = [Comm_Name[j]]*r_Grid
                
                Comm_Name_array = np.reshape(np.array(Comm_Name_array_list),(r_Comm,1))
                
                # Creating Comm Rank Table
                for k in range(len(Comm_Rank_array_list)):
                    
                    if k == 0:
                        
                        Comm_Rank_Array_Combined_array = Comm_Rank_array_list[k]
                        
                    else:
                        
                        Comm_Rank_Array_Combined_array = np.block([Comm_Rank_Array_Combined_array,Comm_Rank_array_list[k]])   
                        
                # Creating Complete Comm Rank Array for Table
                Comm_Rank_Array_Combined_array_ForTable1 = np.hstack((Comm_Name_array,Comm_Rank_Array_Combined_array)) 
                
                # IF ELSE LOOP: For 
                if (j==0):
                    
                    Comm_Rank_Array_Combined_array_ForTable = Comm_Rank_Array_Combined_array_ForTable1
                    
                    RankingMethod_ComputationalTime_array_ForTable = RankingMethod_ComputationalTime_array
                    
                else:
                    
                    Comm_Rank_Array_Combined_array_ForTable = np.vstack((Comm_Rank_Array_Combined_array_ForTable,Comm_Rank_Array_Combined_array_ForTable1)) 
                
                    RankingMethod_ComputationalTime_array_ForTable = np.vstack((RankingMethod_ComputationalTime_array_ForTable,RankingMethod_ComputationalTime_array))
                    
                # Computing Similarity between Comm Nodes Ranking Methods
                Comm_Degree_Pagerank_Simirality = Computing_Ranking_Similarity(Comm_DegreeRank_array[:,0], Comm_PageRank_array[:,0])
                
                Comm_Degree_EigenAnalysis_Simirality = Computing_Ranking_Similarity(Comm_DegreeRank_array[:,0], Comm_EigenAnalysisRank_array[:,0])
                
                Comm_Pagerank_EigenAnalysis_Simirality = Computing_Ranking_Similarity(Comm_PageRank_array[:,0], Comm_EigenAnalysisRank_array[:,0])
                
                # Creating Comm Ranking Similarity Table
                Comm_Rank_Similarity_Table1 = np.hstack((np.reshape(np.array([Comm_Name[j]]),(1,1)),np.reshape(np.array([Comm_Degree_Pagerank_Simirality,Comm_Degree_EigenAnalysis_Simirality,Comm_Pagerank_EigenAnalysis_Simirality]),(1,3))))
                
                # Creating Combined Grid/Comm Ranking Similarity Table
                if (j==0):
                
                    GridComm_Combined_Rank_Similarity_Table = np.vstack((GridComm_Combined_Rank_Similarity_Table,Comm_Rank_Similarity_Table1))
                    
                else:
                    
                    GridComm_Combined_Rank_Similarity_Table = np.vstack((GridComm_Combined_Rank_Similarity_Table,Comm_Rank_Similarity_Table1))
                
    # Creating Table 1: Grid Rank Value Table
    Data_Frame_Table1 = pd.DataFrame(Grid_Rank_Array_Combined_array_ForTable,columns=['Grid Name', 'Deg_Rank','Deg_Value','PR_Rank','PR_Value','EVA_Rank','EVA_Value'])
                     
    # Creating Table 2: Comm Rank Value Table
    Data_Frame_Table2 = pd.DataFrame(Comm_Rank_Array_Combined_array_ForTable,columns=['Comm Name', 'Deg_Rank','Deg_Value','PR_Rank','PR_Value','EVA_Rank','EVA_Value']) 
    
    # Creating Table 3: Ranking Comparison Grid/Comm Table
    Data_Frame_Table3 = pd.DataFrame(GridComm_Combined_Rank_Similarity_Table,columns=['Grid/Comm Name', 'Deg_PR','Deg_EVA','PR_EVA'])   

    # Creating Table 4: Ranking Method Computation Time Comm
    Data_Frame_Table4 = pd.DataFrame(RankingMethod_ComputationalTime_array_ForTable,columns=['Comm Name', 'Deg_Rank','PR_Rank','EVA_Rank'])   


    # Saving Tables in Excel Files
    File_Path = Result_Path+File_Name
    
    Writer = pd.ExcelWriter(File_Path)
        
    Data_Frame_Table1.to_excel(Writer, sheet_name = Sheet1_Name, index = False)
    
    Data_Frame_Table2.to_excel(Writer, sheet_name = Sheet2_Name, index = False)
    
    Data_Frame_Table3.to_excel(Writer, sheet_name = Sheet3_Name, index = False)
    
    Data_Frame_Table4.to_excel(Writer, sheet_name = Sheet4_Name, index = False)
    
    Writer.save()

    return Data_Frame_Table1, Data_Frame_Table2, Data_Frame_Table3

# =============================================================================
# Plot Function
# =============================================================================
def Creating_Plots(Result_Path, Grid_Name, Comm_Name, Rank_Method_Name, GridCommNetworkFailure_Percentage_Vector, Time_Vector, Storage_FrequencyResponse_BaseCase , Storage_FrequencyResponse_GridNodeFailureCase , Storage_FrequencyResponse_CommNodeFailureCase):
    
    # Defining X axis
    X_Time = Time_Vector
    
    X_Bar = ['Base_Case']
    
    X_Bar_Color = ['black']
    
    # FOR LOOP: For Name of the X Axis Data
    for i in range(len(GridCommNetworkFailure_Percentage_Vector)):
        
        X_Bar.append(str(int(GridCommNetworkFailure_Percentage_Vector[i])))
        X_Bar_Color.append('red')           
      
    
    # FOR LOOP: For Creating Grid Frequency Response Under Node Failuer Plot
    for i in range((len(Storage_FrequencyResponse_GridNodeFailureCase))):
        
        # Creating Grid Frequency Base Response Plot

        # Getting Y axis: Base Frequency Data
        Storage_FrequencyResponse_BaseCase1 = Storage_FrequencyResponse_BaseCase[i]
        
        Y_Frequency_Base = Storage_FrequencyResponse_BaseCase1[1]
        
        Y_Frequency_Base = np.reshape(Y_Frequency_Base,(len(Y_Frequency_Base),1))
        
        # Getting Bar Chart: Time Data 
        Y_Time_Base = Storage_FrequencyResponse_BaseCase1[2]    
        
        Y_Dominant_EigenValue_Base = Storage_FrequencyResponse_BaseCase1[3] 
        
        Y_Stability_Status_Base = Storage_FrequencyResponse_BaseCase1[4] 
        
        # FOR LOOP: For Getting Current Ranking Method
        for j in range((len(Storage_FrequencyResponse_GridNodeFailureCase[i]))): 
            
            # Getting Current Ranking Method Frequency Response Data
            Current_Grid_Ranking_Method = Storage_FrequencyResponse_GridNodeFailureCase[i][j]
            
            Current_Comm_Ranking_Method = Storage_FrequencyResponse_CommNodeFailureCase[i][j]
            
            # Initializing Frequency Array for Stack 
            Y_Grid_Frequency_Node_Failure_Array = Y_Frequency_Base
            
            Y_Comm_Frequency_Node_Failure_Array = Y_Frequency_Base
            
            # Initializing Time Array for Stack
            Y_Grid_Time_Node_Failure_Array = np.reshape(np.array([Y_Time_Base]),(1,1))
            
            Y_Comm_Time_Node_Failure_Array = np.reshape(np.array([Y_Time_Base]),(1,1))
            
            Y_Grid_Dominant_EigenValue_Node_Failure_Array = np.reshape(np.array([Y_Dominant_EigenValue_Base]),(1,1))
            
            Y_Comm_Dominant_EigenValue_Node_Failure_Array = np.reshape(np.array([Y_Dominant_EigenValue_Base]),(1,1))

            Y_Grid_Stability_Status_Node_Failure_Array = np.reshape(np.array([Y_Stability_Status_Base]),(1,1))
            
            Y_Comm_Stability_Status_Node_Failure_Array = np.reshape(np.array([Y_Stability_Status_Base]),(1,1))            
                      
            # FOR LOOP: For Getting Node Failure Frequency Response under Current Comm Data and Ranking Method
            for k in range((len(Current_Grid_Ranking_Method))):
                
                Current_Grid_Node_Failure = Current_Grid_Ranking_Method[k]
                
                Current_Comm_Node_Failure = Current_Comm_Ranking_Method[k]
                
                Y_Grid_Frequency_Node_Failure = Current_Grid_Node_Failure[1]
                
                Y_Comm_Frequency_Node_Failure = Current_Comm_Node_Failure[1]
                
                Y_Grid_Time_Node_Failure = Current_Grid_Node_Failure[2]
                
                Y_Comm_Time_Node_Failure = Current_Comm_Node_Failure[2]
                
                Y_Grid_Dominant_EigenValue_Node_Failure = Current_Grid_Node_Failure[3]
                
                Y_Comm_Dominant_EigenValue_Node_Failure = Current_Comm_Node_Failure[3]    
                
                Y_Grid_Stability_Status_Node_Failure = Current_Grid_Node_Failure[4]
                
                Y_Comm_Stability_Status_Node_Failure = Current_Comm_Node_Failure[4]                  
                
                # Reshaping Y                
                Y_Grid_Frequency_Node_Failure = np.reshape(Y_Grid_Frequency_Node_Failure,(len(Y_Grid_Frequency_Node_Failure),1))
                
                Y_Comm_Frequency_Node_Failure = np.reshape(Y_Comm_Frequency_Node_Failure,(len(Y_Comm_Frequency_Node_Failure),1))
                
                Y_Grid_Time_Node_Failure = np.reshape(Y_Grid_Time_Node_Failure,(1,1))
                
                Y_Comm_Time_Node_Failure = np.reshape(Y_Comm_Time_Node_Failure,(1,1))
                
                Y_Grid_Dominant_EigenValue_Node_Failure = np.reshape(Y_Grid_Dominant_EigenValue_Node_Failure,(1,1))
                
                Y_Comm_Dominant_EigenValue_Node_Failure = np.reshape(Y_Comm_Dominant_EigenValue_Node_Failure,(1,1))
                
                Y_Grid_Stability_Status_Node_Failure = np.reshape(Y_Grid_Stability_Status_Node_Failure,(1,1))
                
                Y_Comm_Stability_Status_Node_Failure = np.reshape(Y_Comm_Stability_Status_Node_Failure,(1,1))
                
                # Stacking All Y
                Y_Grid_Frequency_Node_Failure_Array = np.hstack((Y_Grid_Frequency_Node_Failure_Array,Y_Grid_Frequency_Node_Failure))
                
                Y_Comm_Frequency_Node_Failure_Array = np.hstack((Y_Comm_Frequency_Node_Failure_Array,Y_Comm_Frequency_Node_Failure))
                
                Y_Grid_Time_Node_Failure_Array = np.hstack((Y_Grid_Time_Node_Failure_Array,Y_Grid_Time_Node_Failure))
                
                Y_Comm_Time_Node_Failure_Array = np.hstack((Y_Comm_Time_Node_Failure_Array,Y_Comm_Time_Node_Failure))
                
                Y_Grid_Dominant_EigenValue_Node_Failure_Array = np.hstack((Y_Grid_Dominant_EigenValue_Node_Failure_Array,Y_Grid_Dominant_EigenValue_Node_Failure))
                
                Y_Comm_Dominant_EigenValue_Node_Failure_Array = np.hstack((Y_Comm_Dominant_EigenValue_Node_Failure_Array,Y_Comm_Dominant_EigenValue_Node_Failure))
                
                Y_Grid_Stability_Status_Node_Failure_Array = np.hstack((Y_Grid_Stability_Status_Node_Failure_Array,Y_Grid_Stability_Status_Node_Failure))
                
                Y_Comm_Stability_Status_Node_Failure_Array = np.hstack((Y_Comm_Stability_Status_Node_Failure_Array,Y_Comm_Stability_Status_Node_Failure))
                
                
                
            # Plotting Grid Frequency Respose Graph
            r,c = np.shape(Y_Grid_Frequency_Node_Failure_Array)
            
            Plot_File_Name = "\\" + "GridFailureFrequencyResponse_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            Plot_File_Path = Result_Path + Plot_File_Name
            
            Title = "Grid Failure Frequency Response for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])


            #FOR LOOP: Over Each Frequency_Node_Failure
            for k in range(c):
                
                if k == 0:
                    
                    plt.plot(X_Time[0:len(X_Time)-1], Y_Grid_Frequency_Node_Failure_Array[:,k], color = 'black', label = X_Bar[k])
                
                else:
                    
                    plt.plot(X_Time[0:len(X_Time)-1], Y_Grid_Frequency_Node_Failure_Array[:,k], linestyle = 'dashed', label = X_Bar[k])    
            
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency Deviation (rad/s)")
            plt.legend(bbox_to_anchor=(1.05,0.85))
            plt.title(Title)
            plt.tight_layout()
            
            plt.savefig(Plot_File_Path)
            plt.show()

            # Plotting Comm Frequency Respose Graph
            r,c = np.shape(Y_Comm_Frequency_Node_Failure_Array)
            
            Plot_File_Name = "\\" + "CommFailureFrequencyResponse_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            Plot_File_Path = Result_Path + Plot_File_Name
            
            Title = "Comm Failure Frequency Response for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])


            #FOR LOOP: Over Each Frequency_Node_Failure
            for k in range(c):
                
                if k == 0:
                    
                    plt.plot(X_Time[0:len(X_Time)-1], Y_Comm_Frequency_Node_Failure_Array[:,k], color = 'blue', label = X_Bar[k])
                
                else:
                    
                    plt.plot(X_Time[0:len(X_Time)-1], Y_Comm_Frequency_Node_Failure_Array[:,k], linestyle = 'dashed', label = X_Bar[k])    
            
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency Deviation (rad/s)")
            plt.legend(bbox_to_anchor=(1.05,0.85))
            plt.title(Title)
            plt.tight_layout()
            
            plt.savefig(Plot_File_Path)
            plt.show()
            

            # # Plotting Grid Time Respose Graph
            # Plot_File_Name = "\\" + "GridFailureTimeScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            # Plot_File_Path = Result_Path + Plot_File_Name
            
            # #Title = "Grid Failure Time Score for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])
            
            # plt.bar(X_Bar , height = Y_Grid_Time_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
            
            # plt.xlabel("Nodes")
            # plt.ylabel("Time Score")
            # #plt.title(Title)
            
            # plt.savefig(Plot_File_Path)
            # plt.show()

            # # Plotting Comm Time Respose Graph
            # Plot_File_Name = "\\" + "CommFailureTimeScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            # Plot_File_Path = Result_Path + Plot_File_Name
            
            # #Title = "Comm Failure Time Score for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])
            
            # plt.bar(X_Bar , height = Y_Comm_Time_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
            
            # plt.xlabel("Nodes")
            # plt.ylabel("Time Score")
            # #plt.title(Title)
            
            # plt.savefig(Plot_File_Path)
            # plt.show()
            
            
            # # Plotting Grid Dominant Eigen Value Respose Graph
            # Plot_File_Name = "\\" + "GridFailureDominantEigenValueScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            # Plot_File_Path = Result_Path + Plot_File_Name
            
            # #Title = "Grid Failure Dominant Eigen Value for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])
            
            # plt.bar(X_Bar , height = Y_Grid_Dominant_EigenValue_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
            
            # plt.xlabel("Nodes")
            # plt.ylabel("Dominant Eigen Value Ral Part")
            # #plt.title(Title)
            
            # plt.savefig(Plot_File_Path)
            # plt.show()

            # # Plotting Comm Dominant Eigen Value Respose Graph
            # Plot_File_Name = "\\" + "CommFailureDominantEigenValueScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            # Plot_File_Path = Result_Path + Plot_File_Name
            
            # #Title = "Comm Failure Dominant Eigen Value for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])
            
            # plt.bar(X_Bar , height = Y_Comm_Dominant_EigenValue_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
            
            # plt.xlabel("Nodes")
            # plt.ylabel("Dominant Eigen Value Ral Part")
            # #plt.title(Title)
            
            # plt.savefig(Plot_File_Path)
            # plt.show()    
            

            # # Plotting Grid Stability Status Value Respose Graph
            # Plot_File_Name = "\\" + "GridFailureStabilityStatusScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            # Plot_File_Path = Result_Path + Plot_File_Name
            
            # #Title = "Grid Failure Stability Status for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])
            
            # plt.bar(X_Bar , height = Y_Grid_Stability_Status_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
            
            # plt.xlabel("Nodes")
            # plt.ylabel("Stability Status Value")
            # #plt.title(Title)
            
            # plt.savefig(Plot_File_Path)
            # plt.show()

            # # Plotting Comm Stability Status Value Respose Graph
            # Plot_File_Name = "\\" + "CommFailureStabilityStatusScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + "_RankingMethod_" + str(Rank_Method_Name[j]) + ".png"
            # Plot_File_Path = Result_Path + Plot_File_Name
            
            # #Title = "Comm Failure Stability Status for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) + "; Ranking Method: " + str(Rank_Method_Name[j])
            
            # plt.bar(X_Bar , height = Y_Comm_Stability_Status_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
            
            # plt.xlabel("Nodes")
            # plt.ylabel("Stability Status Value")
            # #plt.title(Title)
            
            # plt.savefig(Plot_File_Path)
            # plt.show()              
            
def Creating_Plots1(Result_Path, Grid_Name, Comm_Name, Rank_Method_Name, GridCommNetworkFailure_Percentage_Vector, Time_Vector, Storage_FrequencyResponse_BaseCase , Storage_FrequencyResponse_GridNodeFailureCase , Storage_FrequencyResponse_CommNodeFailureCase):
    
    # Defining X axis
    X_Time = Time_Vector
    
    # X_Bar = ['Base_Case']
    X_axis = np.arange(len(GridCommNetworkFailure_Percentage_Vector))
    
    # X_Bar_Color = ['black']
    X_Tick = []
    
    # FOR LOOP: For Name of the X Axis Data
    for i in range(len(GridCommNetworkFailure_Percentage_Vector)):
        
        X_Tick.append(str(int(GridCommNetworkFailure_Percentage_Vector[i])))
        # X_Bar_Color.append('red')           
      
    
    # FOR LOOP: For Creating Grid Frequency Response Under Node Failuer Plot
    for i in range((len(Storage_FrequencyResponse_GridNodeFailureCase))):
        
        # Creating Grid Frequency Base Response Plot

        # Getting Y axis: Base Frequency Data
        Storage_FrequencyResponse_BaseCase1 = Storage_FrequencyResponse_BaseCase[i]
        
        Y_Frequency_Base = Storage_FrequencyResponse_BaseCase1[1]
        
        Y_Frequency_Base = np.reshape(Y_Frequency_Base,(len(Y_Frequency_Base),1))
        
        # Getting Bar Chart: Time Data 
        Y_Time_Base = Storage_FrequencyResponse_BaseCase1[2]    
        
        Y_Dominant_EigenValue_Base = Storage_FrequencyResponse_BaseCase1[3] 
        
        Y_Stability_Status_Base = Storage_FrequencyResponse_BaseCase1[4] 
        
        # Initialization
        Y_Grid_Time_Node_Failure_Array_Storage = []
        
        Y_Comm_Time_Node_Failure_Array_Storage = []
        
        Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage = []
        
        Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage = []
        
        Y_Grid_Stability_Status_Node_Failure_Array_Storage = []
        
        Y_Comm_Stability_Status_Node_Failure_Array_Storage = []
        
        # FOR LOOP: For Getting Current Ranking Method
        for j in range((len(Storage_FrequencyResponse_GridNodeFailureCase[i]))): 
            
            # Getting Current Ranking Method Frequency Response Data
            Current_Grid_Ranking_Method = Storage_FrequencyResponse_GridNodeFailureCase[i][j]
            
            Current_Comm_Ranking_Method = Storage_FrequencyResponse_CommNodeFailureCase[i][j]
            
            # Initializing Frequency Array for Stack 
            Y_Grid_Frequency_Node_Failure_Array = Y_Frequency_Base
            
            Y_Comm_Frequency_Node_Failure_Array = Y_Frequency_Base
            
            # Initializing Time Array for Stack
            Y_Grid_Time_Node_Failure_Array = np.empty((1,1))
            
            Y_Comm_Time_Node_Failure_Array = np.empty((1,1))
            
            Y_Grid_Dominant_EigenValue_Node_Failure_Array = np.empty((1,1))
            
            Y_Comm_Dominant_EigenValue_Node_Failure_Array = np.empty((1,1))

            Y_Grid_Stability_Status_Node_Failure_Array = np.empty((1,1))
            
            Y_Comm_Stability_Status_Node_Failure_Array = np.empty((1,1))    
                      
            # FOR LOOP: For Getting Node Failure Frequency Response under Current Comm Data and Ranking Method
            for k in range((len(Current_Grid_Ranking_Method))):
                
                Current_Grid_Node_Failure = Current_Grid_Ranking_Method[k]
                
                Current_Comm_Node_Failure = Current_Comm_Ranking_Method[k]
                
                Y_Grid_Frequency_Node_Failure = Current_Grid_Node_Failure[1]
                
                Y_Comm_Frequency_Node_Failure = Current_Comm_Node_Failure[1]
                
                Y_Grid_Time_Node_Failure = Current_Grid_Node_Failure[2]
                
                Y_Comm_Time_Node_Failure = Current_Comm_Node_Failure[2]
                
                Y_Grid_Dominant_EigenValue_Node_Failure = Current_Grid_Node_Failure[3]
                
                Y_Comm_Dominant_EigenValue_Node_Failure = Current_Comm_Node_Failure[3]    
                
                Y_Grid_Stability_Status_Node_Failure = Current_Grid_Node_Failure[4]
                
                Y_Comm_Stability_Status_Node_Failure = Current_Comm_Node_Failure[4]                  
                
                # Reshaping Y                
                Y_Grid_Frequency_Node_Failure = np.reshape(Y_Grid_Frequency_Node_Failure,(len(Y_Grid_Frequency_Node_Failure),1))
                
                Y_Comm_Frequency_Node_Failure = np.reshape(Y_Comm_Frequency_Node_Failure,(len(Y_Comm_Frequency_Node_Failure),1))
                
                Y_Grid_Time_Node_Failure = np.reshape(Y_Grid_Time_Node_Failure,(1,1))
                
                Y_Comm_Time_Node_Failure = np.reshape(Y_Comm_Time_Node_Failure,(1,1))
                
                Y_Grid_Dominant_EigenValue_Node_Failure = np.reshape(Y_Grid_Dominant_EigenValue_Node_Failure,(1,1))
                
                Y_Comm_Dominant_EigenValue_Node_Failure = np.reshape(Y_Comm_Dominant_EigenValue_Node_Failure,(1,1))
                
                Y_Grid_Stability_Status_Node_Failure = np.reshape(Y_Grid_Stability_Status_Node_Failure,(1,1))
                
                Y_Comm_Stability_Status_Node_Failure = np.reshape(Y_Comm_Stability_Status_Node_Failure,(1,1))
                
                # Stacking All Y
                Y_Grid_Frequency_Node_Failure_Array = np.hstack((Y_Grid_Frequency_Node_Failure_Array,Y_Grid_Frequency_Node_Failure))
                
                Y_Comm_Frequency_Node_Failure_Array = np.hstack((Y_Comm_Frequency_Node_Failure_Array,Y_Comm_Frequency_Node_Failure))
                
                Y_Grid_Time_Node_Failure_Array = np.hstack((Y_Grid_Time_Node_Failure_Array,Y_Grid_Time_Node_Failure))
                
                Y_Comm_Time_Node_Failure_Array = np.hstack((Y_Comm_Time_Node_Failure_Array,Y_Comm_Time_Node_Failure))
                
                Y_Grid_Dominant_EigenValue_Node_Failure_Array = np.hstack((Y_Grid_Dominant_EigenValue_Node_Failure_Array,Y_Grid_Dominant_EigenValue_Node_Failure))
                
                Y_Comm_Dominant_EigenValue_Node_Failure_Array = np.hstack((Y_Comm_Dominant_EigenValue_Node_Failure_Array,Y_Comm_Dominant_EigenValue_Node_Failure))
                
                Y_Grid_Stability_Status_Node_Failure_Array = np.hstack((Y_Grid_Stability_Status_Node_Failure_Array,Y_Grid_Stability_Status_Node_Failure))
                
                Y_Comm_Stability_Status_Node_Failure_Array = np.hstack((Y_Comm_Stability_Status_Node_Failure_Array,Y_Comm_Stability_Status_Node_Failure))
                
                
            # Rank Method Storage for Bart Graphs
            Y_Grid_Time_Node_Failure_Array_Storage.append(Y_Grid_Time_Node_Failure_Array)
            
            Y_Comm_Time_Node_Failure_Array_Storage.append(Y_Comm_Time_Node_Failure_Array)
            
            Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage.append(Y_Grid_Dominant_EigenValue_Node_Failure_Array)
            
            Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage.append(Y_Comm_Dominant_EigenValue_Node_Failure_Array)
            
            Y_Grid_Stability_Status_Node_Failure_Array_Storage.append(Y_Grid_Stability_Status_Node_Failure_Array)
            
            Y_Comm_Stability_Status_Node_Failure_Array_Storage.append(Y_Comm_Stability_Status_Node_Failure_Array)
                
        

        # Plotting Grid Time Respose Graph
        Plot_File_Name = "\\" + "GridFailureTimeScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Grid Failure Time Score for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        plt.bar(X_axis - 0.2 , Y_Grid_Time_Node_Failure_Array_Storage[0][0,1:].tolist(), 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Grid_Time_Node_Failure_Array_Storage[1][0,1:].tolist(), 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Grid_Time_Node_Failure_Array_Storage[2][0,1:].tolist(), 0.2, label = 'Eigenvalue')
        
        plt.xticks(X_axis, X_Tick)
        plt.xlabel("Nodes")
        plt.ylabel("Time Score")
        plt.title(Title)
        plt.legend(bbox_to_anchor=(1.05,0.85), loc='center left')
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
    
        # Plotting Comm Time Respose Graph
        Plot_File_Name = "\\" + "CommFailureTimeScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Comm Failure Time Score for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Comm_Time_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Comm_Time_Node_Failure_Array_Storage[0][0,1:].tolist(), 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Comm_Time_Node_Failure_Array_Storage[1][0,1:].tolist(), 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Comm_Time_Node_Failure_Array_Storage[2][0,1:].tolist(), 0.2, label = 'Eigenvalue')
        
        plt.xlabel("Nodes")
        plt.ylabel("Time Score")
        plt.title(Title)
        plt.legend(bbox_to_anchor=(1.05,0.85), loc='center left')
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
        
        
        # Plotting Grid Dominant Eigen Value Respose Graph
        Plot_File_Name = "\\" + "GridFailureDominantEigenValueScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Grid Failure Dominant Eigen Value for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Grid_Dominant_EigenValue_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage[0][0,1:].tolist(), 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage[1][0,1:].tolist(), 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage[2][0,1:].tolist(), 0.2, label = 'Eigenvalue')
        
        plt.xlabel("Nodes")
        plt.ylabel("Dominant Eigen Value Real Part")
        plt.title(Title)
        plt.legend(bbox_to_anchor=(1.05,0.85), loc='center left')
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
    
        # Plotting Comm Dominant Eigen Value Respose Graph
        Plot_File_Name = "\\" + "CommFailureDominantEigenValueScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Comm Failure Dominant Eigen Value for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Comm_Dominant_EigenValue_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage[0][0,1:].tolist(), 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage[1][0,1:].tolist(), 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage[2][0,1:].tolist(), 0.2, label = 'Eigenvalue')
        
        plt.xlabel("Nodes")
        plt.ylabel("Dominant Eigen Value Real Part")
        plt.title(Title)
        plt.legend(bbox_to_anchor=(1.05,0.85), loc='center left')
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()    
        
    
        # Plotting Grid Stability Status Value Respose Graph
        Plot_File_Name = "\\" + "GridFailureStabilityStatusScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Grid Failure Stability Status for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) 
        
        # plt.bar(X_Bar , height = Y_Grid_Stability_Status_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Grid_Stability_Status_Node_Failure_Array_Storage[0][0,1:].tolist(), 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Grid_Stability_Status_Node_Failure_Array_Storage[1][0,1:].tolist(), 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Grid_Stability_Status_Node_Failure_Array_Storage[2][0,1:].tolist(), 0.2, label = 'Eigenvalue')


        plt.xlabel("Nodes")
        plt.ylabel("Stability Status Value")
        plt.title(Title)
        plt.legend(bbox_to_anchor=(1.05,0.85), loc='center left')
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
    
        # Plotting Comm Stability Status Value Respose Graph
        Plot_File_Name = "\\" + "CommFailureStabilityStatusScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) +".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Comm Failure Stability Status for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Comm_Stability_Status_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Comm_Stability_Status_Node_Failure_Array_Storage[0][0,1:].tolist(), 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Comm_Stability_Status_Node_Failure_Array_Storage[1][0,1:].tolist(), 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Comm_Stability_Status_Node_Failure_Array_Storage[2][0,1:].tolist(), 0.2, label = 'Eigenvalue')


        plt.xlabel("Nodes")
        plt.ylabel("Stability Status Value")
        plt.title(Title)
        plt.legend(bbox_to_anchor=(1.05,0.85), loc='center left')
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()               
                        
            
                
                
            
            
    
    
    
    
    
    
