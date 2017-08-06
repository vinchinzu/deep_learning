# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 08:50:57 2017

@author: lvinze
"""


features = np.array([[0.5, -0.2, 0.1]])
X = np.array([[0.5, -0.2, 0.1]])

targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

        # Set number of nodes in input, hidden and output layers.
        input_nodes = inputs.shape[1]
        hidden_nodes = test_w_i_h.shape[1]
        output_nodes = targets.shape[1]

        # Initialize weights
        weights_input_to_hidden = np.random.normal(0.0, input_nodes**-0.5, 
                                       (input_nodes, hidden_nodes))

        weights_hidden_to_output = np.random.normal(0.0, hidden_nodes**-0.5, 
                                       (hidden_nodes, output_nodes))
        lr = 0.01
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        #self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.
        
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(x):
            return 1 / (1 + np.exp(-x)) # Replace 0 with your sigmoid calculation here
        

            x        
    
    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
            Arguments
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            hidden_inputs = np.dot(X,weights_input_to_hidden)   # signals into hidden layer
            hidden_outputs = sigmoid(hidden_inputs) # signals from hidden layer

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.dot(hidden_outputs, weights_hidden_to_output) # signals into final output layer
            final_outputs = sigmoid(final_inputs) # signals from final output layer
            #
            
            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = y - final_outputs	 # Output layer error is the difference between desired target and actual output.

            # TODO: Backpropagated error terms - Replace these values with your calculations.
            output_error_term = error * final_outputs * (1 - final_outputs)
            #check here
            hidden_error_term = np.dot(output_error_term, weights_hidden_to_output)

            # TODO: Calculate the hidden layer's contribution to the error
            weights_hidden_to_output = np.array(weights_hidden_to_output, ndmin=2)

            hidden_error = np.dot(weights_hidden_to_output, output_error_term)

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * x[:,None]
            # Weight step (hidden to output)
            delta_weights_h_o += output_error_term * hidden_outputs

        # TODO: Update the weights - Replace these values with your calculations.
        #learning_rate = lr = learnrate

        weights_hidden_to_output += lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        weights_input_to_hidden += lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        # is this just the same as above? 
        hidden_inputs = np.dot(features.T, weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = sigmoid(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, weights_hidden_to_output)  # signals into final output layer
        final_outputs = sigmoid(final_inputs) # signals from final output layer 
        
        return final_outputs