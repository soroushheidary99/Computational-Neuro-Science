from numpy.lib.shape_base import kron
from Neuron import Neuron
import matplotlib.pyplot as plt
import random
import math 
import numpy as np


class STDP() : 
    
    def __init__(self, neuronPre, neuronPost, Aplus, Aminus, tauT, synapticDefault = .5, temporalChoice = "based on last spike") : 
        self.neuron1 = neuronPre
        self.neuron2 = neuronPost
        self.tauT = tauT
        self.synapticStrengh = synapticDefault
        self.Aplus = Aplus
        self.Aminus = Aminus
        self.temporalChoice = temporalChoice
        self.lastT1 = None
        self.lastT2 = None
        self.synapseHistory = []
        self.synapticChangeHistory = [.5 for i in range(len(neuron_post.I))]
        self.count = 0

    
    def LearnByLastSpike(self, t1, t2) : 
        if(t1 > t2) : 
            deltaW = min(self.Aplus * math.exp(-abs(t1 - t2) * 0.03125 * .8), 2)
        elif(t1 < t2) : 
            deltaW = max(-self.Aminus * math.exp(-abs(t2 - t1) * 0.03125 * .8), -1.5)
        else : 
            deltaW = 0
        
        
        
        self.synapseHistory.append(self.synapticStrengh)

        
        self.synapticStrengh = max(0, self.synapticStrengh + deltaW)
        
        self.count += 1
        for i in range(max(t1, t2), len(self.synapticChangeHistory)) : 
            self.synapticChangeHistory[i] = self.synapticStrengh
            
        


    def LearnByInterval(self, t, precedence) :         
        if(precedence == "post") : 
            a = np.array(self.neuron1.spike_history)
            learnList = list(np.where(a > t - self.tauT))[0]
            
            
        elif(precedence == "pre") :
            a = np.array(self.neuron2.spike_history)
            learnList = list(np.where(a > t - self.tauT))[0]
            
            
        
        if(precedence == "post"):
            for i in learnList : 
                self.LearnByLastSpike(a[i], t)    
        elif(precedence == "pre"):
            for i in learnList : 
                self.LearnByLastSpike(t, a[i])
        
        del a
        
    
    def Learn(self, precedence, t) : 
        if(self.lastT2 != None and self.lastT1 != None) :    
            if(self.temporalChoice == "based on last spike") : 
                self.LearnByLastSpike(self.lastT1, self.lastT2)
            elif(self.temporalChoice == "based on interval") : 
                self.LearnByInterval(t, precedence)
        else: 
            pass
            
            
            
    def Simulate(self) : 
        for t in range(len(self.neuron1.I)) : 
            self.neuron1.Simulate_tick(t)
            if(self.neuron1.just_spiked[t]) :   
                print(t, self.neuron1.just_spiked[t] )
                self.neuron2.I[t + 1] += 1000 * self.synapticStrengh
                self.lastT1 = t  
                self.Learn("pre", t)
                
                
            self.neuron2.Simulate_tick(t)
            if(self.neuron2.just_spiked[t]) :
                print(t, self.neuron1.just_spiked[t] )
                #self.neuron1.I[t + 1] += 1000000
                self.lastT2 = t
                self.Learn("post", t)
                

            
    

   
    def PlotSynapticChange(self) :

        #plt.plot(self.neuron2.U, color = 'r')
        #plt.plot(self.neuron1.U, color = 'b')
        plt.style.use('ggplot')
        a = self.neuron1.spike_history 
        b = []
        c = []
        for i in range(len(a)) : 
            for j in range(500, 900) :
                b.append(a[i])
                c.append(j/1000)  
        plt.scatter(b, c, marker='.', s=20, color='blueviolet')

        del a, b, c
        a = self.neuron2.spike_history 
        b = []
        c = []
        for i in range(len(a)) : 
            for j in range(0, 400) :
                b.append(a[i])
                c.append(j/1000)    
        plt.scatter(b, c, marker='.', s=20, color='deeppink')

        k = self.synapticChangeHistory
        
        plt.plot(k, color = 'maroon')
        plt.legend(['SynapticChange', 'PostNeuron', 'PreNeuron'], loc='upper left')
        plt.show()



I1 = np.arange(0, 100, 0.03125)
for i in range(len(I1)) : 
    if  10 / 0.03125 < i < 30 / 0.03125 or 50 / 0.03125 < i < 65 / 0.03125 : 
        I1[i] = 6000
    else : 
        I1[i] = 0





I2 = np.arange(0, 100, 0.03125)
for i in range(len(I2)) : 
    if  25 / 0.03125 < i < 55 / 0.03125 or 60 / 0.03125 < i < 80 / 0.03125 : 
        I2[i] = 6000
    else : 
        I2[i] = 0

plt.style.use('ggplot')
plt.plot(I2, color='deeppink')
plt.plot(I1, color='blueviolet')
plt.legend(['PreNeuron', 'PostNeuron'], loc='upper left')
plt.show()

neuron_pre = Neuron(I = I1)
neuron_post = Neuron(I = I2)


stdp_simulation = STDP(neuron_pre, neuron_post, Aplus = 8, Aminus = 6, tauT = 1000, synapticDefault=1)
stdp_simulation.Simulate()


stdp_simulation.PlotSynapticChange()
