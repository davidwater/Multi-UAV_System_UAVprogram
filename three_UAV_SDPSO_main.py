import numpy as np
import time
import math
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline
import csv

class SDPSO(object):
    def __init__(self, start, target, v, h, path_1, path_2, path_3, it, ds, Varsize):
        'SDPSO input'
        self.start = start     # UAVs' starting points
        self.target = target   # UAVs' targeting points
        self.v = v             # UAVs' last velocity (it*6)
        self.h = h             # UAVs' heading (it*3)
        self.path_1 = path_1   # UAV1's path (it*2)
        self.path_2 = path_2   # UAV2's path (it*2)
        self.path_3 = path_3   # UAV3's path (it*2)
        self.it = it           # current iterations
        self.total_dist_1 = math.sqrt((start[0,0] - target[0,0]) ** 2 + (start[0,1] - target[0,1]) ** 2)
        self.total_dist_2 = math.sqrt((start[0,2] - target[0,2]) ** 2 + (start[0,3] - target[0,3]) ** 2)
        self.total_dist_3 = math.sqrt((start[0,4] - target[0,4]) ** 2 + (start[0,5] - target[0,5]) ** 2)
        self.dist_max_12 = math.sqrt((start[0,0] - start[0,2]) ** 2 + (start[0,1] - start[0,3]) ** 2)       # maxi. distance between UAV 1 and 2
        self.dist_max_13 = math.sqrt((start[0,0] - start[0,4]) ** 2 + (start[0,1] - start[0,5]) ** 2)       # maxi. distance between UAV 1 and 3
        self.dist_max_23 = math.sqrt((start[0,2] - start[0,4]) ** 2 + (start[0,3] - start[0,5]) ** 2)       # maxi. distance between UAV 2 and 3
        self.ds = ds           # safety distance
        self.VarMin = -3       # velocity constraints
        self.VarMax = 3        # velocity constraints
        self.Varsize = Varsize # 2 * number of UAV
        ' Global message (information of UAVs) '
        self.uav_id = []
        self.uav_type = []
        self.uav_velocity = []
        self.uav_Rmin = []
        self.uav_position = []
        self.depots = []
        'SDPSO parameters'
        self.MaxIt = 10       # Maximum Number of Iterations
        self.nPop_max = 100    # Population Size (Swarm Size)
        self.nPop_min = 30     # Population Size (Swarm Size)
        self.w = 1             # Inertia Weight
        self.wdamp = 0.99      # Inertia Weight Damping Ratio
        self.c1 = 1.5          # Personal Learning Coefficient
        self.c2 = 2.0          # Global Learning Coefficient
        self.w1 = 2          # relative weight factor of cost funtion 1
        self.w2 = 1          # relative weight factor of cost funtion 2
        self.w3 = 1          # relative weight factor of cost funtion 3
        'simulated annealing (SA) parameters'
        self.p = 0             # initial probability of a suboptimal solution being accepted
        self.rand = np.random.rand()
        self.T = 13             # temperature of current system (T=1)
        'acceleration of convergence based on dimensional learning strategy (DLS)' 
        self.count = 0         # non-updating number
        self.m = 2             # Random threshold fo the acceleration of convergence
        'velocity limits'
        self.VelMax = 3
        self.VelMin = -self.VelMax
        'pre-setup matrix for SDPSO'
        # individual particles
        self.Position = []                                                                           # nPop_max*6   
        self.Cost = np.empty([self.nPop_max, 4])                                                     # nPop_max*4 (cost_1, cost_2, cost_3, cost)
        self.Velocity =  []                                                                          # nPop_max*6
        self.Best_Position = np.empty([self.nPop_max, Varsize])                                      # nPop_max*6
        self.Best_Cost = np.empty([self.nPop_max, 1])                                                # nPop_max*1
        # global
        self.GlobalBest_Cost = float('inf') # scalar
        self.pre_GlobalBest_Cost = 0 # scalar
        self.GlobalBest_Position = np.zeros([1,Varsize]) # 1*6

    def iteration(self):
        # collision potential
        if math.sqrt((self.path_1[self.it,0] - self.path_2[self.it,0]) ** 2 + (self.path_1[self.it,1] - self.path_2[self.it,1]) ** 2) < self.ds or \
        math.sqrt((self.path_1[self.it,0] - self.path_3[self.it,0]) ** 2 + (self.path_1[self.it,1] - self.path_3[self.it,1]) ** 2) < self.ds or \
        math.sqrt((self.path_2[self.it,0] - self.path_3[self.it,0]) ** 2 + (self.path_2[self.it,1] - self.path_3[self.it,1]) ** 2) < self.ds:
            # initialization
            self.Position = np.random.uniform(self.VarMin, self.VarMax, [self.nPop_max, self.Varsize])  
            self.Velocity = np.random.uniform(self.VarMin, self.VarMax, [self.nPop_max, self.Varsize])
            for i in range(self.nPop_max):
                ax1 = self.Position[i,0]
                ay1 = self.Position[i,1]
                ax2 = self.Position[i,2]
                ay2 = self.Position[i,3]
                ax3 = self.Position[i,4]
                ay3 = self.Position[i,5]
                # body frame
                vxn1 = self.v[self.it,0] + ax1; vxn1 = max(vxn1, self.VarMin); vxn1 = min(vxn1, self.VarMax) 
                vyn1 = self.v[self.it,1] + ay1; vyn1 = max(vyn1, self.VarMin); vyn1 = min(vyn1, self.VarMax)
                vxn2 = self.v[self.it,2] + ax2; vxn2 = max(vxn2, self.VarMin); vxn2 = min(vxn2, self.VarMax)
                vyn2 = self.v[self.it,3] + ay2; vyn2 = max(vyn2, self.VarMin); vyn2 = min(vyn2, self.VarMax)
                vxn3 = self.v[self.it,4] + ax3; vxn3 = max(vxn3, self.VarMin); vxn3 = min(vxn3, self.VarMax)
                vyn3 = self.v[self.it,5] + ay3; vyn3 = max(vyn3, self.VarMin); vyn3 = min(vyn3, self.VarMax)
                # body frame to world frame
                wvx1 = vxn1 * math.cos(self.h[self.it,0]) - vyn1 * math.sin(self.h[self.it,0]); wvy1 = vxn1 * math.sin(self.h[self.it,0]) + vyn1 * math.cos(self.h[self.it,0])
                wvx2 = vxn2 * math.cos(self.h[self.it,1]) - vyn2 * math.sin(self.h[self.it,1]); wvy2 = vxn2 * math.sin(self.h[self.it,1]) + vyn2 * math.cos(self.h[self.it,1])
                wvx3 = vxn3 * math.cos(self.h[self.it,2]) - vyn3 * math.sin(self.h[self.it,2]); wvy3 = vxn3 * math.sin(self.h[self.it,2]) + vyn3 * math.cos(self.h[self.it,2])
                # world frame waypoints
                xn1 = self.path_1[self.it,0] + wvx1
                yn1 = self.path_1[self.it,1] + wvy1
                xn2 = self.path_2[self.it,0] + wvx2
                yn2 = self.path_2[self.it,1] + wvy2
                xn3 = self.path_3[self.it,0] + wvx3
                yn3 = self.path_3[self.it,1] + wvy3
                # world frame heading
                theta_1 = self.heading(xn1, self.target[0,0], yn1, self.target[0,1]) # new heading
                theta_2 = self.heading(xn2, self.target[0,2], yn2, self.target[0,3]) # new heading
                theta_3 = self.heading(xn3, self.target[0,4], yn3, self.target[0,5]) # new heading
                # calculate cost values
                cost_1 = self.cal_cost_1(xn1, yn1, xn2, yn2, xn3, yn3)
                cost_2 = self.cal_cost_2(xn1, yn1, xn2, yn2, xn3, yn3)
                if self.it == 0:
                    cost_3 = self.cal_cost_3(theta_1, theta_1, theta_2, theta_2, theta_3, theta_3)
                else:
                    cost_3 = self.cal_cost_3(theta_1, self.h[self.it-1, 0], theta_2, self.h[self.it-1, 1], theta_3, self.h[self.it-1, 2])
                
                cost = self.w1*cost_1 + self.w2*cost_2 + self.w3*cost_3
                self.Cost[i,0] = cost_1
                self.Cost[i,1] = cost_2
                self.Cost[i,2] = cost_3
                self.Cost[i,3] = cost
                # update personal best
                self.Best_Position[i,:] = self.Position[i,:]
                self.Best_Cost[i] = self.Cost[i,3]
                #update global best
                if self.Best_Cost[i] < self.GlobalBest_Cost:
                    self.GlobalBest_Cost = self.Best_Cost[i]
                    self.GlobalBest_Position = self.Best_Position[i,:]

            # SDPSO main loop
            for j in range(self.MaxIt):
                    for i in range(self.nPop_max):
                        self.Velocity[i,:] = self.w * self.Velocity[i,:] 
                        + self.c1 * np.random.rand(1,self.Varsize) * (self.Best_Position[i,:] - self.Position[i,:])
                        + self.c2 * np.random.rand(1,self.Varsize) * (self.GlobalBest_Position - self.Position[i,:])

                        # apply velocity limits
                        for k in range(self.Varsize):
                            self.Velocity[i,k] = max(self.Velocity[i,k], self.VelMin)
                            self.Velocity[i,k] = min(self.Velocity[i,k], self.VelMax)

                        # update position
                        self.Position[i,:] = self.Position[i,:] + self.Velocity[i,:] 

                        # evaluation
                        ax1 = self.Position[i,0]
                        ay1 = self.Position[i,1]
                        ax2 = self.Position[i,2]
                        ay2 = self.Position[i,3]
                        ax3 = self.Position[i,4]
                        ay3 = self.Position[i,5]
                        # body frame
                        vxn1 = self.v[self.it,0] + ax1; vxn1 = max(vxn1, self.VarMin); vxn1 = min(vxn1, self.VarMax) 
                        vyn1 = self.v[self.it,1] + ay1; vyn1 = max(vyn1, self.VarMin); vyn1 = min(vyn1, self.VarMax)
                        vxn2 = self.v[self.it,2] + ax2; vxn2 = max(vxn2, self.VarMin); vxn2 = min(vxn2, self.VarMax)
                        vyn2 = self.v[self.it,3] + ay2; vyn2 = max(vyn2, self.VarMin); vyn2 = min(vyn2, self.VarMax)
                        vxn3 = self.v[self.it,4] + ax3; vxn3 = max(vxn3, self.VarMin); vxn3 = min(vxn3, self.VarMax)
                        vyn3 = self.v[self.it,5] + ay3; vyn3 = max(vyn3, self.VarMin); vyn3 = min(vyn3, self.VarMax)
                        # body frame to world frame
                        wvx1 = vxn1 * math.cos(self.h[self.it,0]) - vyn1 * math.sin(self.h[self.it,0]); wvy1 = vxn1 * math.sin(self.h[self.it,0]) + vyn1 * math.cos(self.h[self.it,0])
                        wvx2 = vxn2 * math.cos(self.h[self.it,1]) - vyn2 * math.sin(self.h[self.it,1]); wvy2 = vxn2 * math.sin(self.h[self.it,1]) + vyn2 * math.cos(self.h[self.it,1])
                        wvx3 = vxn3 * math.cos(self.h[self.it,2]) - vyn3 * math.sin(self.h[self.it,2]); wvy3 = vxn3 * math.sin(self.h[self.it,2]) + vyn3 * math.cos(self.h[self.it,2])
                        # world frame waypoints
                        xn1 = self.path_1[self.it,0] + wvx1
                        yn1 = self.path_1[self.it,1] + wvy1
                        xn2 = self.path_2[self.it,0] + wvx2
                        yn2 = self.path_2[self.it,1] + wvy2
                        xn3 = self.path_3[self.it,0] + wvx3
                        yn3 = self.path_3[self.it,1] + wvy3
                        # world frame heading
                        theta_1 = self.heading(xn1, self.target[0,0], yn1, self.target[0,1]) # new heading
                        theta_2 = self.heading(xn2, self.target[0,2], yn2, self.target[0,3]) # new heading
                        theta_3 = self.heading(xn3, self.target[0,4], yn3, self.target[0,5]) # new heading
                        # calculate cost values
                        cost_1 = self.cal_cost_1(xn1, yn1, xn2, yn2, xn3, yn3)
                        cost_2 = self.cal_cost_2(xn1, yn1, xn2, yn2, xn3, yn3)
                        if self.it == 0:
                            cost_3 = self.cal_cost_3(theta_1, theta_1, theta_2, theta_2, theta_3, theta_3)
                        else:
                            cost_3 = self.cal_cost_3(theta_1, self.h[self.it-1, 0], theta_2, self.h[self.it-1, 1], theta_3, self.h[self.it-1, 2])
                        cost = self.w1*cost_1 + self.w2*cost_2 + self.w3*cost_3
                        self.Cost[i,0] = cost_1
                        self.Cost[i,1] = cost_2
                        self.Cost[i,2] = cost_3
                        self.Cost[i,3] = cost

                        # update personal best
                        if self.count > self.m:
                            if self.Cost[i,3] < self.Best_Cost[i]:
                                self.Best_Position[i,:] = self.Position[i,:]
                                self.Best_Cost[i] = self.Cost[i,3]
                                self.count = 0

                                # update global best
                                if self.Best_Cost[i] < self.GlobalBest_Cost:
                                    if self.p > float(self.rand):
                                        self.pre_GlobalBest_Cost = self.GlobalBest_Cost
                                        self.GlobalBest_Cost = self.Best_Cost[i]
                                        self.GlobalBest_Position = self.Best_Position[i,:]
                        
                        # update parameters
                    if j > 0:
                        self.count += 1
                        self.w = self.w * self.wdamp
                        r = (self.MaxIt - j) / self.MaxIt
                        self.T = self.T * r
                        try:
                            self.p = math.exp(-(self.GlobalBest_Cost - self.pre_GlobalBest_Cost)/self.T)
                        except:
                            self.p = math.inf
                        self.m = (j / self.MaxIt) * (self.VarMax - self.VarMin)
        # without collision potential
        else:
            # initialization
            self.Position = np.random.uniform(self.VarMin, self.VarMax, [self.nPop_min, self.Varsize])  
            self.Velocity = np.random.uniform(self.VarMin, self.VarMax, [self.nPop_min, self.Varsize])
            for i in range(self.nPop_min):
                ax1 = self.Position[i,0]
                ay1 = self.Position[i,1]
                ax2 = self.Position[i,2]
                ay2 = self.Position[i,3]
                ax3 = self.Position[i,4]
                ay3 = self.Position[i,5]
                # body frame
                vxn1 = self.v[self.it,0] + ax1; vxn1 = max(vxn1, self.VarMin); vxn1 = min(vxn1, self.VarMax) 
                vyn1 = self.v[self.it,1] + ay1; vyn1 = max(vyn1, self.VarMin); vyn1 = min(vyn1, self.VarMax)
                vxn2 = self.v[self.it,2] + ax2; vxn2 = max(vxn2, self.VarMin); vxn2 = min(vxn2, self.VarMax)
                vyn2 = self.v[self.it,3] + ay2; vyn2 = max(vyn2, self.VarMin); vyn2 = min(vyn2, self.VarMax)
                vxn3 = self.v[self.it,4] + ax3; vxn3 = max(vxn3, self.VarMin); vxn3 = min(vxn3, self.VarMax)
                vyn3 = self.v[self.it,5] + ay3; vyn3 = max(vyn3, self.VarMin); vyn3 = min(vyn3, self.VarMax)
                # body frame to world frame
                wvx1 = vxn1 * math.cos(self.h[self.it,0]) - vyn1 * math.sin(self.h[self.it,0]); wvy1 = vxn1 * math.sin(self.h[self.it,0]) + vyn1 * math.cos(self.h[self.it,0])
                wvx2 = vxn2 * math.cos(self.h[self.it,1]) - vyn2 * math.sin(self.h[self.it,1]); wvy2 = vxn2 * math.sin(self.h[self.it,1]) + vyn2 * math.cos(self.h[self.it,1])
                wvx3 = vxn3 * math.cos(self.h[self.it,2]) - vyn3 * math.sin(self.h[self.it,2]); wvy3 = vxn3 * math.sin(self.h[self.it,2]) + vyn3 * math.cos(self.h[self.it,2])
                # world frame waypoints
                xn1 = self.path_1[self.it,0] + wvx1
                yn1 = self.path_1[self.it,1] + wvy1
                xn2 = self.path_2[self.it,0] + wvx2
                yn2 = self.path_2[self.it,1] + wvy2
                xn3 = self.path_3[self.it,0] + wvx3
                yn3 = self.path_3[self.it,1] + wvy3
                # world frame heading
                theta_1 = self.heading(xn1, self.target[0,0], yn1, self.target[0,1]) # new heading
                theta_2 = self.heading(xn2, self.target[0,2], yn2, self.target[0,3]) # new heading
                theta_3 = self.heading(xn3, self.target[0,4], yn3, self.target[0,5]) # new heading
                # calculate cost values
                cost_1 = self.cal_cost_1(xn1, yn1, xn2, yn2, xn3, yn3)
                cost_2 = self.cal_cost_2(xn1, yn1, xn2, yn2, xn3, yn3)
                if self.it == 0:
                    cost_3 = self.cal_cost_3(theta_1, theta_1, theta_2, theta_2, theta_3, theta_3)
                else:
                    cost_3 = self.cal_cost_3(theta_1, self.h[self.it-1, 0], theta_2, self.h[self.it-1, 1], theta_3, self.h[self.it-1, 2])
                
                cost = self.w1*cost_1 + self.w2*cost_2 + self.w3*cost_3
                self.Cost[i,0] = cost_1
                self.Cost[i,1] = cost_2
                self.Cost[i,2] = cost_3
                self.Cost[i,3] = cost
                # update personal best
                self.Best_Position[i,:] = self.Position[i,:]
                self.Best_Cost[i] = self.Cost[i,3]
                #update global best
                if self.Best_Cost[i] < self.GlobalBest_Cost:
                    self.GlobalBest_Cost = self.Best_Cost[i]
                    self.GlobalBest_Position = self.Best_Position[i,:]

            # SDPSO main loop
            for j in range(self.MaxIt):
                for i in range(self.nPop_min):
                    self.Velocity[i,:] = self.w * self.Velocity[i,:] 
                    + self.c1 * np.random.rand(1,self.Varsize) * (self.Best_Position[i,:] - self.Position[i,:])
                    + self.c2 * np.random.rand(1,self.Varsize) * (self.GlobalBest_Position - self.Position[i,:])

                    # apply velocity limits
                    for k in range(self.Varsize):
                        self.Velocity[i,k] = max(self.Velocity[i,k], self.VelMin)
                        self.Velocity[i,k] = min(self.Velocity[i,k], self.VelMax)

                    # update position
                    self.Position[i,:] = self.Position[i,:] + self.Velocity[i,:]
                    

                    # evaluation
                    ax1 = self.Position[i,0]
                    ay1 = self.Position[i,1]
                    ax2 = self.Position[i,2]
                    ay2 = self.Position[i,3]
                    ax3 = self.Position[i,4]
                    ay3 = self.Position[i,5]
                    # body frame
                    vxn1 = self.v[self.it,0] + ax1; vxn1 = max(vxn1, self.VarMin); vxn1 = min(vxn1, self.VarMax) 
                    vyn1 = self.v[self.it,1] + ay1; vyn1 = max(vyn1, self.VarMin); vyn1 = min(vyn1, self.VarMax)
                    vxn2 = self.v[self.it,2] + ax2; vxn2 = max(vxn2, self.VarMin); vxn2 = min(vxn2, self.VarMax)
                    vyn2 = self.v[self.it,3] + ay2; vyn2 = max(vyn2, self.VarMin); vyn2 = min(vyn2, self.VarMax)
                    vxn3 = self.v[self.it,4] + ax3; vxn3 = max(vxn3, self.VarMin); vxn3 = min(vxn3, self.VarMax)
                    vyn3 = self.v[self.it,5] + ay3; vyn3 = max(vyn3, self.VarMin); vyn3 = min(vyn3, self.VarMax)
                    # body frame to world frame
                    wvx1 = vxn1 * math.cos(self.h[self.it,0]) - vyn1 * math.sin(self.h[self.it,0]); wvy1 = vxn1 * math.sin(self.h[self.it,0]) + vyn1 * math.cos(self.h[self.it,0])
                    wvx2 = vxn2 * math.cos(self.h[self.it,1]) - vyn2 * math.sin(self.h[self.it,1]); wvy2 = vxn2 * math.sin(self.h[self.it,1]) + vyn2 * math.cos(self.h[self.it,1])
                    wvx3 = vxn3 * math.cos(self.h[self.it,2]) - vyn3 * math.sin(self.h[self.it,2]); wvy3 = vxn3 * math.sin(self.h[self.it,2]) + vyn3 * math.cos(self.h[self.it,2])
                    # world frame waypoints
                    xn1 = self.path_1[self.it,0] + wvx1
                    yn1 = self.path_1[self.it,1] + wvy1
                    xn2 = self.path_2[self.it,0] + wvx2
                    yn2 = self.path_2[self.it,1] + wvy2
                    xn3 = self.path_3[self.it,0] + wvx3
                    yn3 = self.path_3[self.it,1] + wvy3
                    # world frame heading
                    theta_1 = self.heading(xn1, self.target[0,0], yn1, self.target[0,1]) # new heading
                    theta_2 = self.heading(xn2, self.target[0,2], yn2, self.target[0,3]) # new heading
                    theta_3 = self.heading(xn3, self.target[0,4], yn3, self.target[0,5]) # new heading
                    # calculate cost values
                    cost_1 = self.cal_cost_1(xn1, yn1, xn2, yn2, xn3, yn3)
                    cost_2 = self.cal_cost_2(xn1, yn1, xn2, yn2, xn3, yn3)
                    if self.it == 0:
                        cost_3 = self.cal_cost_3(theta_1, theta_1, theta_2, theta_2, theta_3, theta_3)
                    else:
                        cost_3 = self.cal_cost_3(theta_1, self.h[self.it-1, 0], theta_2, self.h[self.it-1, 1], theta_3, self.h[self.it-1, 2])
                    cost = self.w1*cost_1 + self.w2*cost_2 + self.w3*cost_3
                    self.Cost[i,0] = cost_1
                    self.Cost[i,1] = cost_2
                    self.Cost[i,2] = cost_3
                    self.Cost[i,3] = cost

                    # update personal best
                    if self.count > self.m:
                        if self.Cost[i,3] < self.Best_Cost[i]:
                            self.Best_Position[i,:] = self.Position[i,:]
                            self.Best_Cost[i] = self.Cost[i,3]
                            self.count = 0

                            # update global best
                            if self.Best_Cost[i] < self.GlobalBest_Cost:
                                self.GlobalBest_Cost = self.Best_Cost[i]
                                self.GlobalBest_Position = self.Best_Position[i,:]
                    
                # update parameters
                if j > 0:
                    self.count += 1
                    self.w = self.w * self.wdamp
                    self.m = (j / self.MaxIt) * (self.VarMax - self.VarMin)

        ax1 = self.GlobalBest_Position[0]
        ay1 = self.GlobalBest_Position[1]
        ax2 = self.GlobalBest_Position[2]
        ay2 = self.GlobalBest_Position[3]
        ax3 = self.GlobalBest_Position[4]
        ay3 = self.GlobalBest_Position[5]
        vxn1 = self.v[self.it,0] + ax1; vxn1 = max(vxn1, self.VarMin); vxn1 = min(vxn1, self.VarMax) 
        vyn1 = self.v[self.it,1] + ay1; vyn1 = max(vyn1, self.VarMin); vyn1 = min(vyn1, self.VarMax)
        vxn2 = self.v[self.it,2] + ax2; vxn2 = max(vxn2, self.VarMin); vxn2 = min(vxn2, self.VarMax)
        vyn2 = self.v[self.it,3] + ay2; vyn2 = max(vyn2, self.VarMin); vyn2 = min(vyn2, self.VarMax)
        vxn3 = self.v[self.it,4] + ax3; vxn3 = max(vxn3, self.VarMin); vxn3 = min(vxn3, self.VarMax)
        vyn3 = self.v[self.it,5] + ay3; vyn3 = max(vyn3, self.VarMin); vyn3 = min(vyn3, self.VarMax)

        return vxn1, vyn1, vxn2, vyn2, vxn3, vyn3, self.GlobalBest_Cost
    
    def heading(self, xn, target_xn, yn, target_yn):
        delta_x = target_xn - xn
        delta_y = target_yn - yn
        theta = np.arctan2(delta_y, delta_x)
        return theta

    def cal_cost_1(self, xn1, yn1, xn2, yn2, xn3, yn3):
        dist_1 = math.sqrt((xn1 - self.target[0,0]) ** 2 + (yn1 - self.target[0,1]) ** 2)
        dist_2 = math.sqrt((xn2 - self.target[0,2]) ** 2 + (yn2 - self.target[0,3]) ** 2)
        dist_3 = math.sqrt((xn3 - self.target[0,4]) ** 2 + (yn3 - self.target[0,5]) ** 2)
        k = 1
        coeff_pun = k * math.exp(1/(dist_1 + dist_2 + dist_3))
        # coeff_pun = (dist_1 + dist_2 + dist_3) / (self.total_dist_1 + self.total_dist_2 + self.total_dist_3)
        cost_1 = (((dist_1 ** 2)/self.total_dist_1)  + ((dist_2 ** 2)/self.total_dist_2) + ((dist_3 ** 2)/self.total_dist_3)) * coeff_pun
        return cost_1
    
    def cal_cost_2(self, xn1, yn1, xn2, yn2, xn3, yn3):
        dist_12 = math.sqrt((xn1 - xn2) ** 2 + (yn1 - yn2) ** 2)
        dist_13 = math.sqrt((xn1 - xn3) ** 2 + (yn1 - yn3) ** 2)
        dist_23 = math.sqrt((xn2 - xn3) ** 2 + (yn2 - yn3) ** 2)
        k = 500
        coeff_pun_12 = k * math.exp(-((dist_12/self.dist_max_12) ** 2))
        coeff_pun_13 = k * math.exp(-((dist_13/self.dist_max_13) ** 2))
        coeff_pun_23 = k * math.exp(-((dist_23/self.dist_max_23) ** 2))
        if dist_12 > self.ds:
            sub_cost_12 = 0
        else:
            sub_cost_12 = (1/dist_12)*coeff_pun_12

        if dist_13 > self.ds:
            sub_cost_13 = 0
        else:
            sub_cost_13 = (1/dist_13)*coeff_pun_13

        if dist_23 > self.ds:
            sub_cost_23 = 0
        else:
            sub_cost_23 = (1/dist_23)*coeff_pun_23
        
        cost_2 = sub_cost_12 + sub_cost_13 + sub_cost_23
        return cost_2
    
    def cal_cost_3(self, theta_1, theta_1_old, theta_2, theta_2_old, theta_3, theta_3_old):
        k = 300
        theta_max = math.pi/3
        coeff_pun_1 = k * math.exp(-(((theta_1 - theta_1_old)/theta_max) ** 2))
        coeff_pun_2 = k * math.exp(-(((theta_2 - theta_2_old)/theta_max) ** 2))
        coeff_pun_3 = k * math.exp(-(((theta_3 - theta_3_old)/theta_max) ** 2))
        
        if abs(theta_1 - theta_1_old) > theta_max:
            sub_cost_1 = theta_1 * coeff_pun_1
        else:
            sub_cost_1 = 0

        if abs(theta_2 - theta_2_old) > theta_max:
            sub_cost_2 = theta_2 * coeff_pun_2
        else:
            sub_cost_2 = 0

        if abs(theta_3 - theta_3_old) > theta_max:
            sub_cost_3 = theta_3 * coeff_pun_3
        else:
            sub_cost_3 = 0

        cost_3 = sub_cost_1 + sub_cost_2 + sub_cost_3
        return cost_3
    
    
def heading(xn, target_xn, yn, target_yn):
    delta_x = target_xn - xn
    delta_y = target_yn - yn
    theta = np.arctan2(delta_y, delta_x)
    return theta

def b2w(vxn1, vyn1, vxn2, vyn2, vxn3, vyn3, h1, h2, h3):
    wvx1 = vxn1 * math.cos(h1) - vyn1 * math.sin(h1); wvy1 = vxn1 * math.sin(h1) + vyn1 * math.cos(h1)
    wvx2 = vxn2 * math.cos(h2) - vyn2 * math.sin(h2); wvy2 = vxn2 * math.sin(h2) + vyn2 * math.cos(h2)
    wvx3 = vxn3 * math.cos(h3) - vyn3 * math.sin(h3); wvy3 = vxn3 * math.sin(h3) + vyn3 * math.cos(h3)
    return wvx1, wvy1, wvx2, wvy2, wvx3, wvy3


def plot_path(it, x1, y1, x2, y2, x3, y3):
    plt.title(f'Iterations: {it}')
    plt.scatter(x1, y1, color = 'red', s=10, label = 'UAV 1')
    plt.scatter(x2, y2, color = 'blue', s=10,label = 'UAV 2')
    plt.scatter(x3, y3, color = 'green', s=10,label = 'UAV 3')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    if it == 0:
        plt.legend(loc = 'upper left')
    plt.pause(0.3)
    return None

def generate_path(start, target, v):
    start_time = time.time()
    path_1 = np.array([[start[0,0], start[0,1]]])
    path_2 = np.array([[start[0,2], start[0,3]]])
    path_3 = np.array([[start[0,4], start[0,5]]])
    h1 = heading(xn=start[0,0], target_xn=target[0,0], yn=start[0,1], target_yn=target[0,1])
    h2 = heading(xn=start[0,2], target_xn=target[0,2], yn=start[0,3], target_yn=target[0,3])
    h3 = heading(xn=start[0,4], target_xn=target[0,4], yn=start[0,5], target_yn=target[0,5])
    # avoid special case i.e. heading angle = 45*n degree
    if (h1 == math.pi/4 or h1 == 0.75*math.pi or h1 == -math.pi/4 or h1 == -0.75*math.pi):
        path_1[0, 0] += 0.01
    elif (h2 == math.pi/4 or h2 == 0.75*math.pi or h2 == -math.pi/4 or h2 == -0.75*math.pi):
        path_2[0, 0] += 0.01
    elif (h3 == math.pi/4 or h3 == 0.75*math.pi or h3 == -math.pi/4 or h3 == -0.75*math.pi):
        path_3[0, 0] += 0.01
    h =np.matrix([h1, h2, h3])

    UAV1_arrive = False
    UAV2_arrive = False
    UAV3_arrive = False
    iteration = 1000
    ds = 10
    Varsize = 6
    it = 0
    dt = 0.4    # update rate
    d1 = 0      # UAV1 moving distance
    d2 = 0      # UAV2 moving distance
    d3 = 0      # UAV3 moving distance
    d_total = 0 # total moving distance
    df = 8    # acceptance distance
    cost = []   # cost value
    
    for it in range (iteration):
        wypt = SDPSO(start, target, v, h, path_1, path_2, path_3, it, ds, Varsize)
        vxn1, vyn1, vxn2, vyn2, vxn3, vyn3, cost_ = wypt.iteration()
        cost_ = np.round(cost_, 4)
        cost = np.append(cost, cost_, axis = 0)
        v = np.append(v, [[vxn1, vyn1, vxn2, vyn2, vxn3, vyn3]], axis = 0)
        # body frame to world frame
        wvx1, wvy1, wvx2, wvy2, wvx3, wvy3 = b2w(vxn1, vyn1, vxn2, vyn2, vxn3, vyn3, h[0,0], h[0,1], h[0,2])
        # world frame path
        path_1_x = np.round(path_1[it,0] + wvx1*dt, 4)
        path_1_y = np.round(path_1[it,1] + wvy1*dt, 4)
        path_2_x = np.round(path_2[it,0] + wvx2*dt, 4)
        path_2_y = np.round(path_2[it,1] + wvy2*dt, 4)
        path_3_x = np.round(path_3[it,0] + wvx3*dt, 4)
        path_3_y = np.round(path_3[it,1] + wvy3*dt, 4)
        path_1 = np.append(path_1, [[path_1_x, path_1_y]], axis = 0)
        path_2 = np.append(path_2, [[path_2_x, path_2_y]], axis = 0)
        path_3 = np.append(path_3, [[path_3_x, path_3_y]], axis = 0)
        #print(f'iteration: {it+1}, UAV1_x: {path_1[it+1, 0]}, UAV1_y: {path_1[it+1, 1]}, UAV2_x: {path_2[it+1, 0]}, UAV2_y: {path_2[it+1, 1]}, cost: {cost[it+1]}')
        # update heading
        h1_ = heading(path_1_x, target[0,0], path_1_y, target[0,1])
        h2_ = heading(path_2_x, target[0,2], path_2_y, target[0,3])
        h3_ = heading(path_3_x, target[0,4], path_3_y, target[0,5])
        h = np.append(h, [[h1, h2, h3]], axis = 0)
        # avoid special case i.e. heading angle = 45*n degree
        if (h1_ == math.pi/4 or h1_ == 0.75*math.pi or h1_ == -math.pi/4 or h1_ == -0.75*math.pi):
            path_1[it+1, 0] += 0.01
        elif (h2_ == math.pi/4 or h2_ == 0.75*math.pi or h2_ == -math.pi/4 or h2_ == -0.75*math.pi):
            path_2[it+1, 0] += 0.01
        elif (h3_ == math.pi/4 or h3_ == 0.75*math.pi or h3_ == -math.pi/4 or h3_ == -0.75*math.pi):
            path_3[it+1, 0] += 0.01

        if (math.sqrt((path_1[it+1, 0] - target[0,0]) ** 2 + (path_1[it+1, 1] - target[0,1]) ** 2) < df):
            path_1[it+1, 0] = target[0,0]
            path_1[it+1, 1] = target[0,1]
            UAV1_arrive = True

        if (math.sqrt((path_2[it+1, 0] - target[0,2]) ** 2 + (path_2[it+1, 1] - target[0,3]) ** 2) < df):
            path_2[it+1, 0] = target[0,2]
            path_2[it+1, 1] = target[0,3]
            UAV2_arrive = True
        
        if (math.sqrt((path_3[it+1, 0] - target[0,4]) ** 2 + (path_3[it+1, 1] - target[0,5]) ** 2) < df):
            path_3[it+1, 0] = target[0,4]
            path_3[it+1, 1] = target[0,5]
            UAV3_arrive = True
        
        # plot_path(it, np.array(path_1[:, 0]), np.array(path_1[:, 1]), np.array(path_2[:, 0]), np.array(path_2[:, 1]))
        # check if UAVs arrive goals
        if UAV1_arrive and UAV2_arrive and UAV3_arrive:
            # plt.show()
            break
            
        # moving distance
        d1 = math.sqrt((path_1[it+1, 0] - path_1[it, 0]) ** 2 + (path_1[it+1, 1] - path_1[it, 1]) ** 2)
        d2 = math.sqrt((path_2[it+1, 0] - path_2[it, 0]) ** 2 + (path_2[it+1, 1] - path_2[it, 1]) ** 2)
        d3 = math.sqrt((path_3[it+1, 0] - path_3[it, 0]) ** 2 + (path_3[it+1, 1] - path_3[it, 1]) ** 2)
        d_total += (d1 + d2 + d3)
        # print(f'Iteration: {it}, Cost value = {cost[it]}')

    print(f'Cost value = {cost[it]}')
    print(f'Process time: {time.time() - start_time} (sec)')
    print(f'total distance = {d_total}')
    
    return path_1, path_2, path_3, h, d_total, cost  

def smooth_path(path_1, path_2, path_3):
    # sort array
    path_1 = path_1[path_1[:,0].argsort()]
    path_2 = path_2[path_2[:,0].argsort()]
    path_3 = path_3[path_3[:,0].argsort()]

    s_1 = path_1.shape[0] * 5
    s_2 = path_2.shape[0] * 10
    s_3 = path_3.shape[0] * 5

    # spline regression
    tck_1 = splrep(path_1[:,0], path_1[:,1], s = s_1)
    tck_2 = splrep(path_2[:,0], path_2[:,1], s = s_2)
    tck_3 = splrep(path_3[:,0], path_3[:,1], s = s_3)

    path_1[:,1] = BSpline(*tck_1)(path_1[:,0])
    path_2[:,1] = BSpline(*tck_2)(path_2[:,0])
    path_3[:,1] = BSpline(*tck_3)(path_3[:,0])

    return path_1, path_2, path_3

if __name__ == "__main__":
    # initial parameters
    xv1 = 0; yv1 = 0; xv2 = 0; yv2 = 0; xv3 = 0; yv3 = 0
    xs1 = -20; ys1 = 20; xg1 = -120; yg1 = 120
    xs2 = -70; ys2 = 20; xg2 = -70; yg2 = 120
    xs3 = -120; ys3 = 20; xg3 = -20; yg3 = 120
    v = np.matrix([xv1, yv1, xv2, yv2, xv3, yv3])
    start = np.matrix([xs1, ys1, xs2, ys2, xs3, ys3])
    target = np.matrix([xg1, yg1, xg2, yg2, xg3, yg3])
    path_1, path_2, path_3, h, d_total, cost = generate_path(start, target, v)

    # save path
    with open('3_UAVs_SDPSO_path.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['UAV1_x', 'UAV1_y', 'UAV2_x', 'UAV2_y', 'UAV3_x', 'UAV3_y'])
        writer.writerows(zip(*[path_1[:,0], path_1[:,1], path_2[:,0], path_2[:,1], path_3[:,0], path_3[:,1]]))

    # smooth path
    path_1_, path_2_, path_3_ = smooth_path(path_1, path_2, path_3)

    # spline regression
    plt.scatter(path_1[:,0], path_1[:,1], color = 'red', s=1, label = 'UAV 1')
    plt.scatter(path_2[:,0], path_2[:,1], color = 'blue', s=1, label = 'UAV 2')
    plt.scatter(path_3[:,0], path_3[:,1], color = 'green', s=1, label = 'UAV 3')
    plt.plot(path_1_[:,0], path_1_[:,1], label = 'UAV 1 spline')
    plt.plot(path_2_[:,0], path_2_[:,1], label = 'UAV 2 spline')
    plt.plot(path_3_[:,0], path_3_[:,1], label = 'UAV 3 spline')
    plt.legend()
    plt.show()