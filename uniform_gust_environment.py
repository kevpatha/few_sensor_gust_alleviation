import math

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

import torch as T
import torch.nn as nn

import math
import time
import serial
import random

import nidaqmx
from nidaqmx.constants import TerminalConfiguration

class mfc_gust_Env(gym.Env):

    def __init__(self, start_volt = 2.5, goal_position=0, steps_per_ep = 200, seed = 0):
        
        T.manual_seed(seed)
        np.random.seed(seed)
 
        # Set up ni communication for pressure taps and MFCs
        self.press_taps = nidaqmx.Task()
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev2/ai0", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev2/ai1", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev2/ai2", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev2/ai3", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev2/ai4", terminal_config = TerminalConfiguration.RSE)
        self.press_taps.ai_channels.add_ai_voltage_chan("Dev2/ai5", terminal_config = TerminalConfiguration.RSE)
        #self.press_taps.ai_channels.add_ai_voltage_chan("Dev2/ai23", terminal_config = TerminalConfiguration.RSE)
        num_taps = 6

        self.load_cell = nidaqmx.Task()
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai0")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai1")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai2")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai3")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai4")
        self.load_cell.ai_channels.add_ai_voltage_chan("Dev3/ai5")
        

        self.mfc_out1 = nidaqmx.Task()
        self.mfc_out1.ao_channels.add_ao_voltage_chan("Dev2/ao0")
        self.mfc_out1.write([start_volt])
        self.mfc_out2 = nidaqmx.Task()
        self.mfc_out2.ao_channels.add_ao_voltage_chan("Dev2/ao1")
        self.mfc_out2.write([start_volt])
        self.mfc_out3 = nidaqmx.Task()
        self.mfc_out3.ao_channels.add_ao_voltage_chan("Dev2/ao3")
        self.mfc_out3.write([start_volt])
        self.new_volt1 = start_volt
        
        # Set up serial communication for gust generator
        self.ser = serial.Serial()
        self.ser.baudrate = 9600
        self.ser.port = 'COM5'
        self.ser.open()
        time.sleep(2)
        
        speed = str('F,C,S1M6000,R')
        self.ser.write(speed.encode())
        
        self.g_done = '^'
        self.g_check = 0
        self.g_rotating = False
        
        self.g_direct = True
        #self.motor1 = str('F,C,I1M-2000,R')
        #self.motor2 = str('F,C,I1M4000,R')
        #self.motor3 = str('F,C,I1M-4000,R')
        #self.motor4 = str('F,C,I1M2000,R')
        #self.ser.write(self.motor2.encode())
        #self.g_done = self.ser.read(self.ser.inWaiting()).decode()
        
        self.max_ep_steps = steps_per_ep
        self.steps_per_ep = steps_per_ep
        
        self.num_steps = 0
        self.first_ep = True
      
        self.min_press = 0
        self.max_press = 5
        self.press_mean = 2.5
        self.press_std = 0.5
        
        self.lift_mean = 0
        self.lift_std = 1 
        self.load_mat = np.array([[-0.91164, 0.22078, -0.71620, -35.41503, 2.10003, 34.48183],
                                  [1.56291, 39.81764, -1.03218, -20.15276,  -0.44775, -20.02389]])
        self.offset = np.array([16.41, -8.06])
        self.lift_rot_mat = np.array([-0.9962, -0.0872])
        self.drag_rot_mat = np.array([-0.0872, 0.9962]) #check this

        
        
        self.min_volt = 0
        self.max_volt = 5
        self.volt_std = 2.5 #was 2.5 post 5
        self.volt_mean = 2.5
        
        
        #self.goal_lift = -3.7
        #print("goal lift: ", self.goal_lift)
        
        self.old_volt = 0
        self.volt1 = 0
        
        #time checking
        self.delt=0.05
        self.dwait = 0
        self.last_dt = 0
        self.prev_time = time.time()
        self.current_time = time.time()
        
       
        
        self.num_taps = num_taps
        num_mfc = 1
        self.num_mfc = num_mfc
        self.action_space = spaces.Discrete(3)
        self.actions1 = np.array([-0.25,0,0.25])
        self.actions2 = np.array([-0.25,0,0.25])
        self.actions3 = np.array([-0.25,0,0.25])
        
            
        self.low_state = np.ones((num_taps+num_mfc,))
        self.low_state[:num_taps] = self.low_state[:num_taps]*self.normalize_press(self.min_press)
        self.low_state[-num_mfc:] = self.low_state[-num_mfc:]*self.normalize_volt(self.min_volt)
        
        self.high_state = np.ones((num_taps+num_mfc,))
        self.high_state[:num_taps] = self.high_state[:num_taps]*self.normalize_press(self.max_press)
        self.high_state[-num_mfc:] = self.high_state[-num_mfc:]*self.normalize_volt(self.max_volt)
           
       
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(num_taps+num_mfc,),  #currently only 7 pressure taps, will change
            dtype=np.float32
        )

        #set goal lift and pressure and initial obs
        self.obs = np.zeros((self.num_taps+self.num_mfc,10))
        self.goal_press = np.zeros((self.num_taps+self.num_mfc,))
        temp_lift = []
        temp_drag = []
        for i in range(10):
            norm_press = self.get_obs()
            #norm_press=self.normalize_press(press_measurements)
            
            time.sleep(0.04)
            self.obs[:self.num_taps,i] = norm_press
            self.obs[-1:,i] = self.volt1
            
            lift, drag = self.get_lift_drag()
            
            temp_lift.append(lift)
            temp_drag.append(drag)
        
        self.goal_lift = np.mean(temp_lift)
        self.goal_press[:self.num_taps] = np.mean(self.obs[:self.num_taps],1)
        self.obs = self.obs-self.goal_press.reshape((-1,1))
        
        print(self.goal_lift)
        print(self.obs[:,-1]-self.goal_press)
        #self.seed()
        
        #self.reset()

    def normalize_volt(self, volt):
        norm_act = (volt-self.volt_mean)/self.volt_std  #/5
        return norm_act
    def denormalize_volt(self, norm_volt):
        act = norm_volt*self.volt_std+self.volt_mean
        return act
    

    def normalize_press(self, press):
        norm_press = (press-self.press_mean)/self.press_std
        return norm_press

    
    def change_gust(self):
        #print('changing gust', self.g_check)        
        if self.g_direct:
            mot_val=int(np.random.rand()*1300+150) #was 2000+750
            if np.random.rand() > 0.5:
                self.motor1 = str('F,C,I1M'+str(mot_val)+',R')
                self.motor2 = str('F,C,I1M'+str(-mot_val)+',R')
            else:
                self.motor1 = str('F,C,I1M'+str(-mot_val)+',R')
                self.motor2 = str('F,C,I1M'+str(mot_val)+',R')
                
            self.ser.write(self.motor1.encode())
            self.g_rotating = True
            #self.g_check+=1
        else:
            self.ser.write(self.motor2.encode())
            self.g_rotationg = True
            #self.g_check+=1
            
        #if self.g_direct:
            #if self.g_check == 0:
                #self.ser.write(self.motor1.encode())
                #self.g_check+=1
            #elif self.g_check == 1:
                #self.ser.write(self.motor2.encode())
                #self.g_check+=1
            #elif self.g_check == 2:
                #self.ser.write(self.motor3.encode())
                #self.g_check += 1
            #elif self.g_check == 3:
                #self.ser.write(self.motor4.encode())
                #self.g_check += 1
        #else:
            #if self.g_check == 0:
                #self.ser.write(self.motor4.encode())
                #self.g_check+=1
            #elif self.g_check == 1:
                #self.ser.write(self.motor3.encode())
                #self.g_check+=1
            #elif self.g_check == 2:
                #self.ser.write(self.motor2.encode())
                #self.g_check += 1
            #elif self.g_check == 3:
                #self.ser.write(self.motor1.encode())
                #self.g_check += 1
            
    def get_obs(self, N_avg=1):
        press = np.array(self.press_taps.read())
        norm_press=self.normalize_press(press) 
        return norm_press
    
    def get_lift_drag(self):
        load_read = np.array(self.load_cell.read())
        force = np.dot(self.load_mat, load_read)-self.offset
        lift = np.dot(force, self.lift_rot_mat)
        drag = np.dot(force, self.drag_rot_mat)
        return lift, drag
    
    def get_reward(self, lift, drag):
        Ker = 10
        Kef = 0.02
        error = self.goal_lift - lift
        #dl = drag/lift
        return -Ker*(error*error)#-Kef*(dl*dl)
    
    def check_table(self):
        self.g_done = self.ser.read(self.ser.inWaiting()).decode()
        if self.g_done == '^':
            self.g_rotating = False
            
    def pause_for_timing(self):
        self.dwait += self.delt - self.last_dt
        while True:
            current_time = time.time()
            dt = current_time - self.prev_time 
            if dt >= self.dwait:
                break
        self.last_dt = dt
        self.prev_time = current_time

    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #done = False
        self.old_volt1 = self.state[-1]
        self.new_volt1 = self.denormalize_volt(self.old_volt1) + self.actions1[action]
        
        if self.new_volt1<= self.min_volt:
            self.new_volt1 = self.min_volt
        elif self.new_volt1 >= self.max_volt:
            self.new_volt1 = self.max_volt
        

            
       # send voltage to mfcs 
        self.mfc_out1.write([self.new_volt1])
        self.mfc_out2.write([self.new_volt1])
        self.mfc_out3.write([self.new_volt1])
        self.volt1 = self.normalize_volt(self.new_volt1)
        self.state[-1] = self.volt1
   
        
        #timing
        self.pause_for_timing()
        
        #checking turn table movement
        #self.check_table()
            
        #if self.num_steps == 0:
         #   self.change_gust()
            
        #if self.num_steps == 150:
         #   self.change_gust()
            
        #if self.g_check == 0:
         #   self.g_done = '^'
            
        
        
        
        norm_press = self.get_obs()-self.goal_press[:self.num_taps] #difference in pressure
        #norm_press=self.normalize_press(press_measurements) 
        new_obs = np.array(norm_press).reshape((1,1,self.num_taps))  
        
      
        self.state[:self.num_taps] = new_obs
        
        state = self.state.reshape(-1,1)
        
        #new_obs = np.append(self.state.reshape(self.num_taps+self.num_mfc,), self.goal_position).reshape(-1,1)
        self.obs = np.append(self.obs, state, axis=1)
        self.obs = np.delete(self.obs, 0, 1)
        
        self._lift = self.lift_
        self._drag = self.drag_
        self.lift_, self.drag_ = self.get_lift_drag()
        lift = (self._lift+self.lift_)/2
        drag = (self._drag+self.drag_)/2
        
        error =  self.goal_lift - lift 
        
        reward = self.get_reward(lift,drag)
        self.num_steps += 1
        if self.num_steps >= self.steps_per_ep:
            self.done = True
            
        return self.obs, reward, self.done, lift, drag

    def reset(self, start_volt = 2.5, goal=0):
        #self.goal_lift = self.get_lift()
        #print(self.goal_lift)
        self.done = False
        self.g_done = '^'
        self.g_check = 0
        #checking turn table movement
        self.check_table()
        self.change_gust()
        
        if self.g_direct:
            self.g_direct = False
            self.steps_per_ep = self.max_ep_steps
            
        else:
            self.g_direct = True
            self.steps_per_ep = self.max_ep_steps/2
            
            self.new_volt1 = start_volt
            self.mfc_out1.write([self.new_volt1])
            self.mfc_out2.write([self.new_volt1])
            self.mfc_out3.write([self.new_volt1])
            self.volt1 = self.normalize_volt(self.new_volt1)
            
            
            press_measurements = self.get_obs()
            temp_lift = []
            temp_drag = []
                
        while self.g_rotating:
            time.sleep(0.04)
            self.check_table()  
        
        if self.g_direct:
            time.sleep(1)
        else:
            time.sleep(0.05)
        self.num_steps = 0
        
        self.obs = np.zeros((self.num_taps+self.num_mfc,10))
        for i in range(10):
            norm_press = self.get_obs()
            #norm_press=self.normalize_press(press_measurements)
            
            time.sleep(0.045)
            self.obs[:self.num_taps,i] = norm_press
            self.obs[-1,i] = self.volt1

            if self.g_direct:
                lift, drag = self.get_lift_drag()
                temp_lift.append(lift)
                temp_drag.append(drag)
        if self.g_direct:
            self.goal_lift = np.mean(temp_lift)
            self.goal_press[:self.num_taps] = np.mean(self.obs[:self.num_taps],1)
            #print(self.goal_lift)
            #print(self.obs[:,-1]-self.goal_press)
        self.obs = self.obs-self.goal_press.reshape((-1,1))
        self.lift_, self.drag_ = self.get_lift_drag()
        self.state = self.obs[:,-1] 
        self.prev_time = time.time()
        self.first_ep = False
        
        return self.obs
    
    def next_goal(self, goal):
        self.goal_lift = goal
    
    def end(self):
        self.new_volt1 = 2.5
        self.mfc_out1.write([self.new_volt1])
        self.mfc_out1.close()
        self.mfc_out2.write([self.new_volt1])
        self.mfc_out2.close()
        self.mfc_out3.write([self.new_volt1])
        self.mfc_out3.close()

        
        
    def close_com(self):
        self.mfc_out1.close()
        self.mfc_out2.close()
        self.mfc_out3.close()
        self.load_cell.close()
        self.press_taps.close()
        self.ser.close()
    
