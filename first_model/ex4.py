# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:47:50 2017

@author: lvinze
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


                  



class NeuralNetwork(object):

    def sigmoid(self, x):
        return ( 1 / (1+np.exp(-x)))  

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x : sigmoid(x)  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        self.activation_function = self.sigmoid

    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targenetworkts: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.dot(hidden_inputs, self.weights_hidden_to_output) # signals into final output layer
            final_outputs = self.activation_function(final_inputs) # signals from final output layer

            print ("--------------------------")
            print ("features")
            print (features)
            print ("targets")
            print (targets)
            print ("X")
            print (X)
            print ("y")
            print (y)

            print ("self.weights_input_to_hidden")
            print (self.weights_input_to_hidden)
            print ("weights_hidden_to_output")
            print (self.weights_hidden_to_output)


            print ("--------------------------")
            print ("hidden_inputs")
            print (hidden_inputs)
            print ("hidden_outputs")
            print (hidden_outputs)

            print ("final_inputs")
            print (final_inputs)
            print ("final_outputs")
            print (final_outputs)
          
            print ("--------------------------")
            
            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = y - final_outputs # Output layer error is the difference between desired target and actual output.

            print ("error")
            print (error)

             # TODO: Backpropagated error terms - Replace these values with your calculations.
            output_error_term = error * final_outputs * (1 - final_outputs)

            print ("output_error_term")
            print (output_error_term)     

            print ("self.weights_hidden_to_output")
            print (self.weights_hidden_to_output)     

            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)

            print ("hidden_error")
            print (hidden_error)

            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

            print ("hidden_error_term")
            print (hidden_error_term)
          
            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:, None]
            print ("delta_weights_i_h")
            print (delta_weights_i_h)
            # Weight step (hidden to output)
            delta_weights_h_o += output_error_term * hidden_outputs[:, None]
            print ("delta_weights_h_o")
            print (delta_weights_h_o)

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records# update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr  * delta_weights_i_h / n_records# update input-to-hidden weights with gradient descent step

        print ("self.weights_hidden_to_output")
        print (self.weights_hidden_to_output)

        print ("weights_input_to_hidden")
        print (self.weights_input_to_hidden)
 
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs





import unittest

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    
    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()
        
        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output, 
                                    np.array([[ 0.37275328], 
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, -0.20185996], 
                                              [0.39775194, 0.50074398], 
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)