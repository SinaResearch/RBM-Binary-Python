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
        #print(vector)
        
        if(index==0):
            training_data=np.array([vector],float)
        else:
            training_data = np.vstack([training_data,vector])
        
    return training_data,num_visible

def add1Backslash(text):
    return text.replace("\\", "\\\\")

def file_validity(File_name): # name of the input file.
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
        
        # Initialize a weight matrix, of dimensions (num_visible x num_hidden), using
        # a Gaussian distribution with mean 0 and standard deviation 0.1.
        self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)    
        # Insert weights for the bias units into the first row and first column.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)

    def train(self, data, max_epochs = 1000):
                
        """
        Train the machine.
        Parameters
        ----------
        data: A matrix where each row is a training example consisting of the states of visible units.    
        """
        
        errorRate=""
        num_examples = data.shape[0]
        
        # Insert bias units of 1 into the first column.
        data = np.insert(data, 0, 1, axis = 1)
        
        for epoch in range(max_epochs):      
            # Clamp to the data and sample from the hidden units. 
            # (This is the "positive CD phase", aka the reality phase.)
            pos_hidden_activations = np.dot(data, self.weights)      
            pos_hidden_probs = self._logistic(pos_hidden_activations)
            pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
            # Note that we're using the activation *probabilities* of the hidden states, not the hidden states       
            # themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
            # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
            pos_associations = np.dot(data.T, pos_hidden_probs)
            
            # Reconstruct the visible units and sample again from the hidden units.
            # (This is the "negative CD phase", aka the daydreaming phase.)
            neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
            neg_visible_probs = self._logistic(neg_visible_activations)
            neg_visible_probs[:,0] = 1 # Fix the bias unit.
            neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
            neg_hidden_probs = self._logistic(neg_hidden_activations)
            # Note, again, that we're using the activation *probabilities* when computing associations, not the states 
            # themselves.
            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
            
            # Update weights.
            self.weights += self.learning_rate * ((pos_associations - neg_associations) / num_examples)
            
            error = np.sum((data - neg_visible_probs) ** 2)
            errorRate=str(errorRate)+"Epoch "+str(epoch)+": error is "+ str(error)+"\n"
#           print("Epoch %s: error is %s" % (epoch, error))
        return errorRate

    def run_visible(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get a sample of the hidden units.
        
        Parameters
        ----------
        data: A matrix where each row consists of the states of the visible units.
        
        Returns
        -------
        hidden_states: A matrix where each row consists of the hidden units activated from the visible
        units in the data matrix passed in.
        """
        
        num_examples = data.shape[0]
        
        # Create a matrix, where each row is to be the hidden units (plus a bias unit)
        # sampled from a training example.
        hidden_states = np.ones((num_examples, self.num_hidden + 1))
        
        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)
        
        # Calculate the activations of the hidden units.
        hidden_activations = np.dot(data, self.weights)
        # Calculate the probabilities of turning the hidden units on.
        hidden_probs = self._logistic(hidden_activations)
        # Turn the hidden units on with their specified probabilities.
        hidden_states[:,:] = hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
        # Always fix the bias unit to 1.
        # hidden_states[:,0] = 1
        
        # Ignore the bias units.
        hidden_states = hidden_states[:,1:]
        return hidden_states
    
    # TODO: Remove the code duplication between this method and `run_visible`?
    
    def run_hidden(self, data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get a sample of the visible units.
        Parameters
        ----------
        data: A matrix where each row consists of the states of the hidden units.
        Returns
        -------
        visible_states: A matrix where each row consists of the visible units activated from the hidden
        units in the data matrix passed in.
        """
        
        num_examples = data.shape[0]
        
        # Create a matrix, where each row is to be the visible units (plus a bias unit)
        # sampled from a training example.
        visible_states = np.ones((num_examples, self.num_visible + 1))
        
        # Insert bias units of 1 into the first column of data.
        data = np.insert(data, 0, 1, axis = 1)
        
        # Calculate the activations of the visible units.
        visible_activations = np.dot(data, self.weights.T)
        # Calculate the probabilities of turning the visible units on.
        visible_probs = self._logistic(visible_activations)
        # Turn the visible units on with their specified probabilities.
        visible_states[:,:] = visible_probs > np.random.rand(num_examples, self.num_visible + 1)
        # Always fix the bias unit to 1.
        # visible_states[:,0] = 1
        
        # Ignore the bias units.
        visible_states = visible_states[:,1:]
        return visible_states
    
    def daydream(self, num_samples):
        """
        Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        (where each step consists of updating all the hidden units, and then updating all of the visible units),
        taking a sample of the visible units at each step.
        Note that we only initialize the network *once*, so these samples are correlated.
        Returns
        -------
        samples: A matrix, where each row is a sample of the visible units produced while the network was
        daydreaming.
        """
        
        # Create a matrix, where each row is to be a sample of of the visible units 
        # (with an extra bias unit), initialized to all ones.
        samples = np.ones((num_samples, self.num_visible + 1))
        
        # Take the first sample from a uniform distribution.
        samples[0,1:] = np.random.rand(self.num_visible)
        
        # Start the alternating Gibbs sampling.
        # Note that we keep the hidden units binary states, but leave the
        # visible units as real probabilities. See section 3 of Hinton's
        # "A Practical Guide to Training Restricted Boltzmann Machines"
        # for more on why.
        for i in range(1, num_samples):
            visible = samples[i-1,:]
            
            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(visible, self.weights)      
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            # Turn the hidden units on with their specified probabilities.
            hidden_states = hidden_probs > np.random.rand(self.num_hidden + 1)
            # Always fix the bias unit to 1.
            hidden_states[0] = 1
            
            # Recalculate the probabilities that the visible units are on.
            visible_activations = np.dot(hidden_states, self.weights.T)
            visible_probs = self._logistic(visible_activations)
            visible_states = visible_probs > np.random.rand(self.num_visible + 1)
            samples[i,:] = visible_states
        
        # Ignore the bias units (the first column), since they're always set to 1.
        return samples[:,1:]        
      
    def _logistic(self, x):
        return 1.0 / (1 + np.exp(-x))
#----------------------------MAIN----------------------------
training_fileName=input("Training set filename: ")
test_fileName=input("Test set filename: ")
epoch_num=input("Number of epochs: ")
num_hidden=int(input("Number of hidden units:"))
 
training_fileName_val=file_validity(training_fileName)
test_fileName_val=file_validity(test_fileName)

if(training_fileName_val and test_fileName_val):# Train/-trainingSet.txt Test/-testSet.txt
    os.makedirs('RBM-Edwin', exist_ok=True)
    
    T_outputname="RBM-Edwin//RBM "+"Epoch "+epoch_num+" "+(training_fileName.split("/")[-1])
    t_outputname="RBM-Edwin//RBM "+"Epoch "+epoch_num+" "+(test_fileName.split("/")[-1])
    
    T_resultStream=codecs.open(T_outputname,"w","utf-8")
    t_resultStream=codecs.open(t_outputname,"w","utf-8")
     
    training_data,num_visible=vectorisation(add1Backslash(training_fileName))
    user,tra=vectorisation(add1Backslash(test_fileName))
     
    print("%d visible classes and %d hidden classes."%(num_visible,num_hidden))
     
    if __name__ == '__main__':
        r = RBM(num_visible, num_hidden)
        T_resultStream.write(r.train(training_data, max_epochs = 1000))
        T_resultStream.write("Weights are:\n")
        T_resultStream.write(str(r.weights))
        
        t_resultStream.write(str(r.run_visible(user)))
        
else:
    print("ERROR: Check file names!")
    
print("Done------------------------------")
