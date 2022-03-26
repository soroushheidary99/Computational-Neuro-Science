import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import numpy as np
import random
import math

class LIF  :

    def __init__(self, I, R=10, tau_m=8, treshold=-45, dspike=5, U_rest=-79, U_reset=-65, ref_period=0, ref_time=0, total_t=100, dt=0.03125, initail_t=0, make_it_pretty=True) : 
        self.I = I
        self.R = R /1000
        self.U_rest = U_rest 
        self.U_reset = U_reset 
        self.tau_m = tau_m
        self.ref_time = ref_time
        self.ref_period = ref_period 
        self.treshold = treshold 
        self.dspike = dspike 
        self.total_t = total_t
        self.dt = dt
        self.initail_t = initail_t
        
        self.tau_to_i = max(1, int(self.ref_period // (self.dt )))
        self.T = np.arange(0, total_t + dt, dt)
        self.Create_U()


        self.sizes = [[1, 5, 3, 1]] if make_it_pretty else [[1, 1, 1, 1]]
        self.colors = [['royalblue','azure', 'lightskyblue'], ['red',  'gold', 'orange'], ['limegreen', 'white', 'springgreen']]
        
    
    def Create_U(self) : 
        self.U = np.zeros(len(self.T))
        for i in range(len(self.U)) : 
            self.U[i] = self.U_reset
        self.U[0] = self.U_rest


    def Simulate(self) :
        i=1
        while i <= len(self.T) - 1 : 

            delta_term = (-(self.U[i-1] - self.U_rest) + (self.R * self.I[i-1])  ) / (self.tau_m) * self.dt 
            self.U[i] = self.U[i-1] + delta_term

            self.U[i] = min(self.U[i-1] + delta_term, self.treshold)
            #self.U[i] = max(self.U[i-1] + delta_term, self.U_rest)

            # Spiking 
            if self.U[i] >= self.treshold : 
                self.U[i] += self.dspike
                i += self.tau_to_i
            i+=1


    def Create_FI_Curve(self) : 
        lengh = len(self.U)
        self.fi = []

        for j in range(1000, 8000):
            self.Create_U()
            self.I = np.array([j]*lengh)
            spike_times = []
            spikes = 0
            i=1
            while i <= len(self.T) - 2 : 
                if spikes == 2 : 
                    break
                delta_term = (-(self.U[i-1] - self.U_rest) + (self.R * self.I[i-1])  ) / (self.tau_m) * self.dt 
                self.U[i] = self.U[i-1] + delta_term

                # Spiking 
                if self.U[i] >= self.treshold : 
                    self.U[i] += self.dspike
                    i += self.tau_to_i
                    spikes += 1
                    spike_times.append(self.T[i])
                i+=1
            if len(spike_times) > 1 : 
                freq = 1/(spike_times[1] - spike_times[0]) 
                self.fi.append(freq)
            else : 
                self.fi.append(0)


    def Plot_I(self) : 
        sns.set_style('dark')
        plt.figure(figsize=(12,4))
        plot = plt.plot(self.T, self.I, color = self.colors[1][0], linewidth=self.sizes[0][0], label='Input Current')
        pe = [path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[1][2], linewidth=self.sizes[0][1], alpha=1), path_effects.Normal(),
              path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[1][1], linewidth=self.sizes[0][2], alpha=1), path_effects.Normal()  ]
        plt.setp(plot, path_effects=pe)
        plt.legend(loc=4)
        plt.grid()
        plt.title('I')
        plt.ylabel('Input Current')
        plt.xlabel('Time')
        plt.savefig('Desktop/LIF21.png', dpi=300)
        #plt.show()
     
    def Plot_U(self) : 
        sns.set_style('dark')
        plt.figure(figsize=(12,4))
        plot = plt.plot(self.T, self.U, color = self.colors[0][0], linewidth=self.sizes[0][0], label='Voltage')
        plt.plot([0,self.total_t], [self.treshold, self.treshold], color='gold', linestyle='-.', label='treshold Voltage')
        pe = [path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[0][2], linewidth=self.sizes[0][1], alpha=1), path_effects.Normal(),
              path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[0][1], linewidth=self.sizes[0][2], alpha=1), path_effects.Normal()  ]
        plt.setp(plot, path_effects=pe)
        plt.legend(loc=4)
        plt.grid()
        plt.title('U')
        plt.ylabel('Voltage')
        plt.xlabel('Time')
        plt.savefig('Desktop/LIF22.png', dpi=300)
        #plt.show()
    
    def Plot_FI(self) : 
        sns.set_style('dark')
        plt.figure(figsize=(12,4))
        plot = plt.plot(range(1000, 8000), self.fi, color = self.colors[2][0], linewidth=self.sizes[0][0], label='Frequency')
        pe = [path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[2][2], linewidth=self.sizes[0][1], alpha=1), path_effects.Normal(),
              path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[2][1], linewidth=self.sizes[0][2], alpha=1), path_effects.Normal()  ]
        plt.setp(plot, path_effects=pe)
        plt.legend(loc=4)
        plt.grid()
        plt.title('FI')
        plt.ylabel('Frequency')
        plt.xlabel('Input Current')
        plt.savefig('Desktop/LIF23.png', dpi=300)
        #plt.show()
    





class AELIF(LIF) : 

    def __init__(self, theta_rh=-58, sharpness=1, a=0.01, b=500, tau_w=100, *args, **kwargs) :
        LIF.__init__(self, *args, **kwargs)
        self.theta_rh = theta_rh 
        self.sharpness = sharpness  # why we need this again ?!
        self.a = a * 32
        self.b = b * 32
        self.tau_w = tau_w
        self.derak = 0 
        self.W = np.zeros(len(self.U))    
    
    def Simulate(self) :
        i=1
        while i <= len(self.T) - 1 : 
            expo_term = - (self.U[i-1] - self.U_rest) + self.sharpness * math.exp( (self.U[i-1] - self.theta_rh)/self.sharpness ) 
            self.W[i] = ((self.a * (self.U[i-1] - self.U_rest) - self.W[i-1]) + (self.b * self.tau_w * self.derak)) * (self.dt / self.tau_w) + self.W[i-1]
            delta_term =  ( expo_term + (self.R * self.I[i-1])  - self.R*self.W[i]  ) / (self.tau_m) * self.dt 
            self.derak = 0
            
            self.U[i] = min(self.U[i-1] + delta_term, self.treshold)
            #self.U[i] = max(self.U[i-1] + delta_term, self.U_rest)

            # Spiking 
            if self.U[i] >= self.treshold : 
                self.derak = 1
                self.W[i+1] = self.W[i] # a bug
                self.U[i] += self.dspike
                for k in range(i + 1, i+self.tau_to_i +1) : 
                    self.W[k] = ((self.a * (self.U[k-1] - self.U_rest) - self.W[k-1])) * (self.dt / self.tau_w) + self.W[k-1]
                i += self.tau_to_i
            i+=1
    
    
    def Plot_W(self) : 
        sns.set_style('dark')
        plt.figure(figsize=(12,4))
        plot = plt.plot(self.T, self.W, color = self.colors[2][0], linewidth=self.sizes[0][0], label='adaptive change')
        pe = [path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[2][2], linewidth=self.sizes[0][1], alpha=1), path_effects.Normal(),
              path_effects.SimpleLineShadow(offset=(0, 0), shadow_color=self.colors[2][1], linewidth=self.sizes[0][2], alpha=1), path_effects.Normal()  ]
        plt.setp(plot, path_effects=pe)
        plt.legend(loc=4)
        plt.grid()
        plt.title('W')
        plt.ylabel('Adaptive Current')
        plt.xlabel('Time')
        plt.savefig('Desktop/LIF23.png', dpi=300)
        #plt.show()




        
I = np.arange(0, 100.03125, 0.03125)

## LIF 1 sin    
#for i in range(len(I)) : 
#    I[i] = 4000 * (math.sin(I[i]) + 0.9) 

## LIF 2 idk_what
#for i in range(len(I)) : 
#   I[i] = i*i*i/3000000

# LIF 3 god_knows_what
#for i in range(len(I)) : 
#  I[i] = ( 2500 - (50 - I[i]) ** (2) ) * 3


######################################################### AELIF Examples #############################################


# ## AELIF Doc eg 1 
# for i in range(len(I)) : 
#     I[i] = 0 if I[i]<10 else 3000
# settings_for_eg1 = "I=I, R=10, tau_m=8, ref_period=0, treshold=-50, dspike=10, U_rest=-70, U_reset=-65 ,a=0.01, b=500, theta_rh=-58"



# # AELIF Doc eg 2 
# for i in range(len(I)) : 
#     if 10<=I[i]<20 : 
#         I[i] = 2000
#     elif 30<=I[i]<40 : 
#         I[i] = 5000
#     elif 50<=I[i]<60 : 
#         I[i] = 7000
#     else : 
#         I[i] = 0 
# settings_for_eg2 = "I=I, R=10, tau_m=8, ref_period=2, treshold=-40, dspike=5, U_rest=-70, U_reset=-65 ,a=0.01, b=500, theta_rh=-45"


# #LIF 3 god_knows_what
# for i in range(len(I)//2) : 
#  I[i] = ( 2500 - (25 - I[i]) ** (2) ) * 3
# for i in range(len(I)//2, len(I)) : 
#  I[i] = ( 2500 - (75 - I[i]) ** (2) ) * 3
# #settings is same as eg1


# #LIF 3 god_knows_what
# for i in range(len(I)//2) : 
#  I[i] = ( 2500 - (25 - I[i]) ** (2) ) * 3
# for i in range(len(I)//2, len(I)) : 
#  I[i] = ( 2500 - (75 - I[i]) ** (2) ) * 3
# #settings is same as eg1



# # #LIF 4 god_knows_what
# from scipy import stats
# I -= 50
# I = stats.norm.pdf(I, 0, 5) * 100000
# # #settings is same as eg1 with ref per of 1



# # #LIF 4 god_knows_what
for i in range(len(I)) : 
    if I[i]%10 == 0 :
        I[i] = 500 * i
# # #settings is same as eg1 with ref per of 1


neuron_1 = AELIF(I=I, R=10, tau_m=8, ref_period=1, treshold=-50, dspike=10, U_rest=-70, U_reset=-65
                 ,a=0.01, b=500, theta_rh=-58)
neuron_1.Simulate()
neuron_1.Plot_I()
neuron_1.Plot_U()
#neuron_1.Create_FI_Curve()
#neuron_1.Plot_FI()
neuron_1.Plot_W()
print('Done !')







#%% 


# %%
