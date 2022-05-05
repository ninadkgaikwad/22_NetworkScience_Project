%% Checking Linear Systems Matrix ODE Performance

clear all;
clc;
close all;


%% Loading Linear System Microgrid Closed Loop System Matrix
load('A_Mat.mat');

%% Initialization
time_vector = 0:0.001:60;

Deg_Deviation = 0.5;

High_x_ini = 2*pi*(Deg_Deviation/360);

Low_x_ini = -2*pi*(Deg_Deviation/360);

%% Eigen Value Analysis
e = eig(A);

%% Initial Condition
[r,c] = size(A);

Time_Len = length(time_vector);

x_ini = Low_x_ini + (High_x_ini-Low_x_ini)*rand(c,1);

x = zeros(r,Time_Len);

u = zeros(Time_Len,r);

%% Creating Linear System

A_Sys = A;

B_Sys = zeros(r,r);

C_Sys = eye(r);

D_Sys = zeros(r,r);

System = ss(A_Sys, B_Sys, C_Sys, D_Sys);

%% Simulating System

y = lsim(System,u,time_vector,x_ini');

%% Plotting

figure(1)
hold on
for ii=1:r
    
    plot(time_vector',y(:,ii));
    
end
hold off
