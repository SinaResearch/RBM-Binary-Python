from __future__ import print_function
import numpy as np
import codecs
import os.path
from test.test_typechecks import Integer

def eliminate_space(list_input):
    list_input_temp=list()
    for i in list_input:
        if not i=="":
            list_input_temp.append(i)
    return list_input_temp

def vectorisation(inputname):
    T_inputSream=(codecs.open(inputname,"r","utf-8")).read()
    
    T_inputVectors=T_inputSream.split('\n')
    T_inputVectors=eliminate_space(T_inputVectors)
    num_visible=len(eliminate_space(T_inputVectors[0].split(',')))
    
    for index in range(len(T_inputVectors)):
        var=eliminate_space(T_inputVectors[index].split(','))
        
        vector=np.array( var, dtype=np.float64)        
        if(index==0):
            training_data=np.array([vector],float)
        else:
            training_data = np.vstack([training_data,vector])
        
    return training_data,num_visible

def add1Backslash(text):
    return text.replace("\\", "\\\\")

def file_validity(File_name):
    validity=False

    if(os.path.isfile(add1Backslash(File_name))):
        validity=True
    else:
        validity=False
        
    return validity

class RBM:
    def __init__(self, num_visible, num_hidden, learning_rate = 0.1):
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)  
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def train(self, data, max_epochs = 1000):
        errorRate=""
        num_examples = data.shape[0]
        
        data = np.insert(data, 0, 1, axis = 1)
        
        for epoch in range(max_epochs):      
            pos_hidden_activations = np.dot(data, self.weights)      
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            pos_associations = np.dot(data.T, pos_hidden_probs)
            
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)
            
            error = np.sum((data - neg_visible_probs) ** 2)
            errorRate=str(errorRate)+"Epoch "+str(epoch)+": error is "+ str(error)+"\n"
        return errorRate

    def run_visible(self, data):
        num_examples = data.shape[0]
        hidden_states = np.ones((num_examples, self.num_hidden + 1))
        data = np.insert(data, 0, 1, axis = 1)
        hidden_activations = np.dot(data, self.weights)
        hidden_probs = self._logistic(hidden_activations)
        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        
        hidden_states = hidden_states[:,1:]
        return hidden_states
    
    
    def run_hidden(self, data):        
        num_examples = data.shape[0]
        visible_states = np.ones((num_examples, self.num_visible + 1))
        data = np.insert(data, 0, 1, axis = 1)
        
        visible_activations = np.dot(data, self.weights.T)
        visible_probs = self._logistic(visible_activations)
        visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        visible_states = visible_states[:,1:]
        return visible_states
    
    def sampler(self, num_samples):
        samples = np.ones((num_samples, self.num_visible + 1))
        samples[0,1:] = np.random.rand(self.num_visible)
        for i in range(1, num_samples):
            visible = samples[i-1,:]
            
            hidden_activations = np.dot(visible, self.weights)      
            hidden_probs = self._logistic(hidden_activations)
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            hidden_states[0] = 1
            
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i,:] = visible_states
        return samples[:,1:]        
      
    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))