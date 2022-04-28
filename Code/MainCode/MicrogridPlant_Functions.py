# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:41:10 2022

@author: ninad gaikwad and sajjad u. mahmud
"""

# =============================================================================
# Import Required Modules
# =============================================================================

# External Modules
import numpy as np

# Custom Modules
import MicrogridController_Functions as MCF


# =============================================================================
# Computing Linear Dynamics Derivative of Microgrid System
# =============================================================================
def Compute_Microgrid_LinearDynamics_Derivative(x,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type):
    
    # Initializing Derivative Vector
    Derivative = np.zeros((l+p+p,))
    
    # IF ELIF LOOP: For Node Failure Type
    if ((NodeFailure_Type==0) or (NodeFailure_Type==2)): # No Failure Case OR Comm Node Failure     
        
        # FOR LOOP: over each state node for Computing Derivative
        for i in range(l+p+p):
            
            # Getting Current Node
            CurrentNode = i
            
            # IF ELIF LOOP: For nodes segregated as Angle of Inverter Based DGs, Angle of Synchronous Machine based DGs and Angular Speed of Synchronous Machine based DGs
            if (CurrentNode<=(l-1)): # Angle of Inverter Based DGs
            
                # FOR LOOP: Over each Grid Network Node Connection
                for j in range(l+p):
                    
                    # Computing Derivative
                    Derivative[i] = Derivative[i] - ((Adj_p[i,j])*(x[i]-x[j]))
                               
                # Correcting Derivative for Control Command
                Derivative[i] = Derivative[i] + u[i] 
                
                # Correcting Derivative for G
                Derivative[i] = Derivative[i] + (G*x[i])                 
            
            elif (CurrentNode>(l-1+p)): # Angular Speed of Synchronous Machine based DGs 
            
                # FOR LOOP: Over each Grid Network Node Connection
                for j in range(l+p):

                    # Correcting index i
                    ii = i-p
                    
                    # Computing Derivative
                    Derivative[i] = Derivative[i] - ((1/M[i-l-p])*(Adj_p[ii,j])*(x[ii]-x[j]))
                               
                # Correcting Derivative for Control Command
                Derivative[i] = Derivative[i] + ((1/M[i-l-p])*u[i])
                
                # Correcting Derivative for Angle of Synchronous Machine
                Derivative[i] = Derivative[i] - (z*x[i])              
            
            else: # Angle of Synchronous Machine based DGs 

                # Computing Derivative
                Derivative[i] = x[i+p]                    
        
    elif (NodeFailure_Type==1): # Grid Node Failure 
        
        # FOR LOOP: over each state node for Computing Derivative
        for i in range(l+p+p):
            
            # Getting Current Node
            CurrentNode = i

            # IF ELSE LOOP: For Current Node to be Grid Failure Node
            if ((CurrentNode == FailureNode) or ((CurrentNode-p) == FailureNode)):

                # Computing Derivative
                Derivative[i] = 0                      
                    
        
            else:
            
                # IF ELIF LOOP: For nodes segregated as Angle of Inverter Based DGs, Angle of Synchronous Machine based DGs and Angular Speed of Synchronous Machine based DGs
                if (CurrentNode<=(l-1)): # Angle of Inverter Based DGs
                
                    # FOR LOOP: Over each Grid Network Node Connection
                    for j in range(l+p):
                        
                        # IF LOOP: For continuing over Failure Node
                        if (j == FailureNode):
                            
                            continue                        
                        
                        # Computing Derivative
                        Derivative[i] = Derivative[i] - ((Adj_p[i,j])*(x[i]-x[j]))
                                   
                    # Correcting Derivative for Control Command
                    Derivative[i] = Derivative[i] + u[i] 
                    
                    # Correcting Derivative for G
                    Derivative[i] = Derivative[i] + (G*x[i])                 
                
                elif (CurrentNode>(l-1+p)): # Angular Speed of Synchronous Machine based DGs 
                
                    # FOR LOOP: Over each Grid Network Node Connection
                    for j in range(l+p):

                        # IF LOOP: For continuing over Failure Node
                        if (j == FailureNode):
                            
                            continue                            

                        # Correcting index i
                        ii = i-p
                        
                        # Computing Derivative
                        Derivative[i] = Derivative[i] - ((1/M[i-l-p])*(Adj_p[ii,j])*(x[ii]-x[j]))
                                   
                    # Correcting Derivative for Control Command
                    Derivative[i] = Derivative[i] + ((1/M[i-l-p])*u[i]) 
                    
                    # Correcting Derivative for Angle of Synchronous Machine
                    Derivative[i] = Derivative[i] - (z*x[i])              
                
                else: # Angle of Synchronous Machine based DGs 
    
                    # Computing Derivative
                    Derivative[i] = x[i+p] 
                
    return Derivative
    
# =============================================================================
# Computing Nonlinear Dynamics Derivative of Microgrid System
# =============================================================================
def Compute_Microgrid_NonlinearDynamics_Derivative(x,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type):
    
    # Initializing Derivative Vector
    Derivative = np.zeros((l+p+p,))
    
    # IF ELIF LOOP: For Node Failure Type
    if ((NodeFailure_Type==0) or (NodeFailure_Type==2)): # No Failure Case OR Comm Node Failure     
        
        # FOR LOOP: over each state node for Computing Derivative
        for i in range(l+p+p):
            
            # Getting Current Node
            CurrentNode = i
            
            # IF ELIF LOOP: For nodes segregated as Angle of Inverter Based DGs, Angle of Synchronous Machine based DGs and Angular Speed of Synchronous Machine based DGs
            if (CurrentNode<=(l-1)): # Angle of Inverter Based DGs
            
                # FOR LOOP: Over each Grid Network Node Connection
                for j in range(l+p):
                    
                    # Computing Derivative
                    Derivative[i] = Derivative[i] - ((Adj_p[i,j])*np.sin(x[i]-x[j]))
                               
                # Correcting Derivative for Control Command
                Derivative[i] = Derivative[i] + u[i] 
                
                # Correcting Derivative for G
                Derivative[i] = Derivative[i] + (G*np.sin(x[i]))                 
            
            elif (CurrentNode>(l-1+p)): # Angular Speed of Synchronous Machine based DGs 
            
                # FOR LOOP: Over each Grid Network Node Connection
                for j in range(l+p):

                    # Correcting index i
                    ii = i-p
                    
                    # Computing Derivative
                    Derivative[i] = Derivative[i] - ((1/M[i-l-p])*(Adj_p[ii,j])*np.sin(x[ii]-x[j]))
                               
                # Correcting Derivative for Control Command
                Derivative[i] = Derivative[i] + ((1/M[i-l-p])*u[i]) 
                
                # Correcting Derivative for Angle of Synchronous Machine
                Derivative[i] = Derivative[i] - (z*x[i])              
            
            else: # Angle of Synchronous Machine based DGs 

                # Computing Derivative
                Derivative[i] = x[i+p]                    
        
    elif (NodeFailure_Type==1): # Grid Node Failure 
        
        # FOR LOOP: over each state node for Computing Derivative
        for i in range(l+p+p):
            
            # Getting Current Node
            CurrentNode = i

            # IF ELSE LOOP: For Current Node to be Grid Failure Node
            if ((CurrentNode == FailureNode) or ((CurrentNode-p) == FailureNode)):

                # Computing Derivative
                Derivative[i] = 0                      
                    
        
            else:
            
                # IF ELIF LOOP: For nodes segregated as Angle of Inverter Based DGs, Angle of Synchronous Machine based DGs and Angular Speed of Synchronous Machine based DGs
                if (CurrentNode<=(l-1)): # Angle of Inverter Based DGs
                
                    # FOR LOOP: Over each Grid Network Node Connection
                    for j in range(l+p):
                        
                        # IF LOOP: For continuing over Failure Node
                        if (j == FailureNode):
                            
                            continue                        
                        
                        # Computing Derivative
                        Derivative[i] = Derivative[i] - ((Adj_p[i,j])*np.sin(x[i]-x[j]))
                                   
                    # Correcting Derivative for Control Command
                    Derivative[i] = Derivative[i] + u[i] 
                    
                    # Correcting Derivative for G
                    Derivative[i] = Derivative[i] + (G*np.sin(x[i]))                 
                
                elif (CurrentNode>(l-1+p)): # Angular Speed of Synchronous Machine based DGs 
                
                    # FOR LOOP: Over each Grid Network Node Connection
                    for j in range(l+p):

                        # IF LOOP: For continuing over Failure Node
                        if (j == FailureNode):
                            
                            continue                            

                        # Correcting index i
                        ii = i-p
                        
                        # Computing Derivative
                        Derivative[i] = Derivative[i] - ((1/M[i-l-p])*(Adj_p[ii,j])*np.sin(x[ii]-x[j]))
                                   
                    # Correcting Derivative for Control Command
                    Derivative[i] = Derivative[i] + ((1/M[i-l-p])*u[i]) 
                    
                    # Correcting Derivative for Angle of Synchronous Machine
                    Derivative[i] = Derivative[i] - (z*x[i])              
                
                else: # Angle of Synchronous Machine based DGs 
    
                    # Computing Derivative
                    Derivative[i] = x[i+p] 
                
    return Derivative
    
# =============================================================================
# Computing Next Step Using Runge-Kutta 4th Order ODE Method
# =============================================================================
def Compute_NextStep_RungeKutta_4thOrder(x,u,M,z,T,G,Adj_p,PlantDynamics_Type, l, p,TimeDelta, FailureNode, NodeFailure_Type):
    
    # IF ELIF LOOP: For Plant Dynamics Type to be used
    if (PlantDynamics_Type==1): # Linear Dynamics
    
        # Calling custom function to compute Runge-Kutta 4th Order ODE Method elements    
        k1 = Compute_Microgrid_LinearDynamics_Derivative(x,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
        x1 = x + (k1*(TimeDelta/2))
        
        k2 = Compute_Microgrid_LinearDynamics_Derivative(x1,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
        x2 = x + (k2*(TimeDelta/2))
        
        k3 = Compute_Microgrid_LinearDynamics_Derivative(x2,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
        x3 = x + (k3*TimeDelta)
        
        k4 = Compute_Microgrid_LinearDynamics_Derivative(x3,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
    elif (PlantDynamics_Type==2): # Nonlinear Dynamics
            
        # Calling custom function to compute Runge-Kutta 4th Order ODE Method elements    
        k1 = Compute_Microgrid_NonlinearDynamics_Derivative(x,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
        x1 = x + (k1*(TimeDelta/2))
        
        k2 = Compute_Microgrid_NonlinearDynamics_Derivative(x1,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
        x2 = x + (k2*(TimeDelta/2))
        
        k3 = Compute_Microgrid_NonlinearDynamics_Derivative(x2,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
        x3 = x + (k3*TimeDelta)
        
        k4 = Compute_Microgrid_NonlinearDynamics_Derivative(x3,u,M,z,T,G,Adj_p, l, p, FailureNode, NodeFailure_Type)
        
    # Computing Next Step using Runge-Kutta 4th Order ODE Method
    m = (1/6)*(k1) + (1/3)*(k2) + (1/3)*(k3) + (1/6)*(k4) 
    
    x_next = x + m*TimeDelta
    
    # Return next step
    return x_next
            
    

# =============================================================================
# Microgrid Plant Function
# =============================================================================
def Microgrid_Plant_TimeSimulation(x_Ini,M,z,T,G,k,Adj_p,Adj_d,FailureNode,Time_Vector,TimeDelta,NodeFailure_Type,N,p,l,MasterNode,PlantDynamics_Type):
    
    # Getting Time Vector Length
    Time_Vector_Len = Time_Vector.size
    
    # Initializing State Vector
    x = np.zeros((Time_Vector_Len+1,N+p))    
      
    # Initializing State Vector with Initial State vector
    x[0,:] = x_Ini
    
    # Initializing Control Command Vector
    u = np.zeros((Time_Vector_Len,N+p))    
    
    # FOR LOOP: over each element in Time Vector
    for i in range(Time_Vector_Len):
        
        # Getting Current State Vector
        x_0 = x[i,:]
            
        # Calling custom function for computing control command vector
        u_0_Vec = MCF.Compute_Controller_Command(x_0,Adj_d,k,MasterNode, l, p, FailureNode, NodeFailure_Type)
                
        # Storing Control Command Vector
        u[i,:] = u_0_Vec   
         
        # Calling custom function for computing next state vector
        x_next_Vec = Compute_NextStep_RungeKutta_4thOrder(x_0,u_0_Vec,M,z,T,G,Adj_p,PlantDynamics_Type, l, p,TimeDelta, FailureNode, NodeFailure_Type)    
            
        # Storing next state Vector
        x[i+1,:] = x_next_Vec 
        
        # Getting Angle Respone Matrix               
        AngleResponse_Matrix = x[:,range(N)]
        
        # Debugger
        print(i)
            
    # Return Angle Response Matrix
    return AngleResponse_Matrix 
