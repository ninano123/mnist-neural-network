#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NeuralNetwork define
class neuralNetwork:
    
    #init NeuralNetwork
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set node number of ipnut / hidden / output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #select sigmoid function to activation function
        self.activation_function = lambda x: scipy.special.expit(x)
            
        #learning rate
        self.lr = learningrate
        
        #wih, who are matrix of weight
        #weight is discribe like w_i_j
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        

        pass
    
    #train NeuralNetwork
    def train(self, inputs_list, targets_list):
        #transmit input_list matrix 
        inputs = np.array(inputs_lists, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #calculate input/output for hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #calculate input/output for final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        #error
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        #weight update between hidden / final layer
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0- final_outputs)), np,transpose(hidden_outputs))
        #weight update between input / hidden layer
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0- hidden_outputs)), np,transpose(inputs))
        
        pass
    
    #query NeuralNetwork
    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        #input / output for hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_ouputs = self.activation_function(hidden_inputs)
        
        #input / output for final layer
        final_inputs = np.dot(self.who, hidden_ouputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
        pass


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.special

#number of input / hidden / output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

#learning rate is 0.3
learning_rate = 0.3

#create instance of neural network
n = neuralNetwork(input_nodes,
                 hidden_nodes,
                 output_nodes,
                 learning_rate)


# In[3]:


n.query([1.0,0.5,-1.5])


# In[ ]:




