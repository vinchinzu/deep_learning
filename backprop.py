# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 11:28:20 2017

@author: lvinze
"""

import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate output error
error = None

# TODO: Calculate error term for output layer
output_error_term = None

# TODO: Calculate error term for hidden layer
hidden_error_term = None

# TODO: Calculate change in weights for hidden layer to output layer
delta_w_h_o = None

# TODO: Calculate change in weights for input layer to hidden layer
delta_w_i_h = None

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
