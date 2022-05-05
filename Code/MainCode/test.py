# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:43:03 2022

@author: sajjaduddin.mahmud
"""

import numpy as np 
import matplotlib.pyplot as plt 
  
# X = ['Group A','Group B','Group C','Group D']
# Ygirls = [10,20,20,40]
# Zboys = [20,30,25,30]
# Wnoone= [1,4,8,15]
  
# X_axis = np.arange(len(X))
  
# plt.bar(X_axis - 0.2, Ygirls, 0.2, label = 'Girls')
# plt.bar(X_axis + 0.0, Zboys, 0.2, label = 'Boys')
# plt.bar(X_axis + 0.2, Wnoone, 0.2, label = 'NOOne')
  
# plt.xticks(X_axis, X)
# plt.xlabel("Groups")
# plt.ylabel("Number of Students")
# plt.title("Number of Students in each group")
# plt.legend()
# plt.show()

def Creating_Plots(Result_Path, Grid_Name, Comm_Name, Rank_Method_Name, GridCommNetworkFailure_Percentage_Vector, Time_Vector, Storage_FrequencyResponse_BaseCase , Storage_FrequencyResponse_GridNodeFailureCase , Storage_FrequencyResponse_CommNodeFailureCase):
    
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
        
        plt.bar(X_axis - 0.2 , Y_Grid_Time_Node_Failure_Array_Storage.tolist()[0], 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Grid_Time_Node_Failure_Array_Storage.tolist()[1], 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Grid_Time_Node_Failure_Array_Storage.tolist()[2], 0.2, label = 'Eigenvalue')
        
        plt.xticks(X_axis, X_Tick)
        plt.xlabel("Nodes")
        plt.ylabel("Time Score")
        plt.title(Title)
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
    
        # Plotting Comm Time Respose Graph
        Plot_File_Name = "\\" + "CommFailureTimeScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Comm Failure Time Score for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Comm_Time_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Comm_Time_Node_Failure_Array_Storage.tolist()[0], 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Comm_Time_Node_Failure_Array_Storage.tolist()[1], 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Comm_Time_Node_Failure_Array_Storage.tolist()[2], 0.2, label = 'Eigenvalue')
        
        plt.xlabel("Nodes")
        plt.ylabel("Time Score")
        plt.title(Title)
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
        
        
        # Plotting Grid Dominant Eigen Value Respose Graph
        Plot_File_Name = "\\" + "GridFailureDominantEigenValueScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Grid Failure Dominant Eigen Value for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Grid_Dominant_EigenValue_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage.tolist()[0], 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage.tolist()[1], 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Grid_Dominant_EigenValue_Node_Failure_Array_Storage.tolist()[2], 0.2, label = 'Eigenvalue')
        
        plt.xlabel("Nodes")
        plt.ylabel("Dominant Eigen Value Ral Part")
        plt.title(Title)
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
    
        # Plotting Comm Dominant Eigen Value Respose Graph
        Plot_File_Name = "\\" + "CommFailureDominantEigenValueScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Comm Failure Dominant Eigen Value for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Comm_Dominant_EigenValue_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage.tolist()[0], 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage.tolist()[1], 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Comm_Dominant_EigenValue_Node_Failure_Array_Storage.tolist()[2], 0.2, label = 'Eigenvalue')
        
        plt.xlabel("Nodes")
        plt.ylabel("Dominant Eigen Value Ral Part")
        plt.title(Title)
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()    
        
    
        # Plotting Grid Stability Status Value Respose Graph
        Plot_File_Name = "\\" + "GridFailureStabilityStatusScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) + ".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Grid Failure Stability Status for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i]) 
        
        # plt.bar(X_Bar , height = Y_Grid_Stability_Status_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Grid_Stability_Status_Node_Failure_Array_Storage.tolist()[0], 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Grid_Stability_Status_Node_Failure_Array_Storage.tolist()[1], 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Grid_Stability_Status_Node_Failure_Array_Storage.tolist()[2], 0.2, label = 'Eigenvalue')


        plt.xlabel("Nodes")
        plt.ylabel("Stability Status Value")
        plt.title(Title)
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()
    
        # Plotting Comm Stability Status Value Respose Graph
        Plot_File_Name = "\\" + "CommFailureStabilityStatusScore_Grid_" + Grid_Name + "_Comm_" + str(Comm_Name[i]) +".png"
        Plot_File_Path = Result_Path + Plot_File_Name
        
        Title = "Comm Failure Stability Status for Grid: " + Grid_Name + "; Comm: " + str(Comm_Name[i])
        
        # plt.bar(X_Bar , height = Y_Comm_Stability_Status_Node_Failure_Array.tolist()[0] , color = X_Bar_Color)
        plt.bar(X_axis - 0.2 , Y_Comm_Stability_Status_Node_Failure_Array_Storage.tolist()[0], 0.2, label = 'Degree')
        plt.bar(X_axis + 0.0 , Y_Comm_Stability_Status_Node_Failure_Array_Storage.tolist()[1], 0.2, label = 'PageRank' )
        plt.bar(X_axis + 0.2 , Y_Comm_Stability_Status_Node_Failure_Array_Storage.tolist()[2], 0.2, label = 'Eigenvalue')


        plt.xlabel("Nodes")
        plt.ylabel("Stability Status Value")
        plt.title(Title)
        plt.tight_layout()
        
        plt.savefig(Plot_File_Path)
        plt.show()              
            
            
                        
            
                
                
            
            
    
    
    
    
    
    
