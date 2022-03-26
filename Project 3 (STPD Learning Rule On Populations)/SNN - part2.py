import numpy as np
import matplotlib.pyplot as plt
import math
from Neuron import Neuron
plt.style.use('ggplot')


class SNN() : 
    def __init__(self, I, pre_neurons=[], post_neurons=[]) : 
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        
        self.synapsDictionary = {}

        self.I = I
        self.I_zero = [0 for i in range(len(I[0]))]
        
    def CreateNeurons(self) : 
        for i in range(10) : 
            if i != 4 and i != 9 : 
                self.pre_neurons.append(Neuron(I=self.I[i%5]))
            else : 
                self.pre_neurons.append(Neuron(I=self.I_zero))
        
        for i in range(2) : 
            self.post_neurons.append(Neuron(I=self.I_zero))

        #self.post_neurons[0].last_spike=100
        #self.post_neurons[0].last_spike=100

    def CreateSynapses(self) : 
        for i in range(len(self.pre_neurons)) : 
            if(i <= 4) :
                self.synapsDictionary[i] = [2 * (i%5+1)/1.5, 0]
            else : 
                self.synapsDictionary[i] = [0, 2 * (i%5+1)/1.5]

    def SimulateByTick(self, t) : 
        for n in range(len(self.pre_neurons)) : 
            self.pre_neurons[n].Simulate_tick(t)
            if self.pre_neurons[n].last_spike == t :
                if(n<=4):  
                    self.STDP(n, 0, self.pre_neurons[n].last_spike, self.post_neurons[0].last_spike)
                    self.post_neurons[0].I[t+1] += 80000 * (self.synapsDictionary[n][0]) 
                if(n>5):  
                    self.STDP(n, 1, self.pre_neurons[n].last_spike, self.post_neurons[1].last_spike)
                    self.post_neurons[1].I[t+1] += 80000 * (self.synapsDictionary[n][1]) 


        for j in range(2) : 
            self.post_neurons[j].Simulate_tick(t)
            if self.post_neurons[j].just_spiked : 
                for n in range(len(self.pre_neurons)) : 
                    self.STDP(n, j, self.pre_neurons[n].last_spike, self.post_neurons[j].last_spike)
                    

    
    def STDP(self, n1, n2, t1, t2) : 
        deltaW = 0
        if(self.synapsDictionary[n1][n2] != 0) : 
            if(t1 != None and t2 != None):
                if(t1 > t2) : 
                    deltaW = min(2 * math.exp(-abs(t1 - t2) * 0.03125 * .8), 0.2)
                elif(t1 < t2) : 
                    deltaW = max(-1.5 * math.exp(-abs(t2 - t1) * 0.03125 * .8), -0.15)

            self.synapsDictionary[n1][n2] += deltaW * 100
            if(self.synapsDictionary[n1][n2] <= 0) : 
                self.synapsDictionary[n1][n2] = 0.01



    # there is a chance of learning besid 0 1 0 1


I1 = np.arange(0, 100 + 0.03125, 0.03125)
I2 = np.arange(0, 100 + 0.03125, 0.03125)
I3 = np.arange(0, 100 + 0.03125, 0.03125)
I4 = np.arange(0, 100 + 0.03125, 0.03125)

for i in range(len(I1)) : 
    I1[i] = 0
    I2[i] = 0
    I3[i] = 0
    I4[i] = 0
    
for i in range(len(I1)) : 
    if 0 < i < 750 :
        I1[i] = 2500
    elif 750 < i < 1500 : 
        I2[i] = 2500
    elif 1500 < i < 2250 : 
        I3[i] = 2500
    elif 2250 < i < 3000 : 
        I4[i] = 2500


# plt.plot(I1)
# plt.plot(I2)
# plt.plot(I3)
# plt.plot(I4)
# plt.title("Is")
# plt.legend([ *(['I' + str(i + 1) + " and I" + str(i + 6) for i in range(0, 4)])])
# plt.show()



# I should be inputed like this: [I] not this: I
simpleSNN = SNN(I = [I1, I2, I3, I4])
simpleSNN.CreateNeurons()
simpleSNN.CreateSynapses()


for epoch in range(5) : 
    for i in range(len(I1) - 1) : 
        simpleSNN.SimulateByTick(i)

    for i in range(5, 9) :
        plt.plot(simpleSNN.pre_neurons[i].U)

    plt.plot(simpleSNN.post_neurons[1].U)
    plt.title(str(epoch + 1) + "th Epoch")
    plt.legend([ *(['Pre' + str(i + 1) for i in range(5, 9)]), "Post 2" ])
    plt.savefig('/users/sorou/Desktop/epoch' + str(epoch + 1) + "pttr22", bbox_inches='tight')
    plt.clf()



print(simpleSNN.synapsDictionary)





    
    
    
    

    