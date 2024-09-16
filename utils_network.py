'''
First implemented: 01/25/2018
  > For survival analysis on longitudinal dataset
By CHANGHEE LEE

Modifcation List:
	- 08/07/2018: weight regularization for FC_NET is added
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import GlorotUniform  # Xavier initializer in TF2
from tensorflow.keras.regularizers import l2  # Example regularizer
from tensorflow.keras.layers import GRUCell, LSTMCell, RNN, Dropout

### CONSTRUCT MULTICELL FOR MULTI-LAYER RNNS
def create_rnn_cell(num_units, num_layers, keep_prob, RNN_type):
    '''
        GOAL         : Create a multi-cell (including a single cell) to construct a multi-layer RNN
        num_units    : Number of units in each layer
        num_layers   : Number of layers in MulticellRNN
        keep_prob    : Keep probability [0, 1] (if None, dropout is not employed)
        RNN_type     : Either 'LSTM' or 'GRU'
    '''
    cells = []

    for _ in range(num_layers):
        if RNN_type == 'GRU':
            cell = GRUCell(num_units)
        elif RNN_type == 'LSTM':
            cell = LSTMCell(num_units)

        # Apply dropout to each cell, if keep_prob < 1.0
        if keep_prob < 1.0:
            cell = Dropout(1 - keep_prob)(cell)

        cells.append(cell)

    # Stack the RNN cells into a multi-layer RNN using RNN wrapper
    multi_rnn = RNN(cells, return_sequences=True, return_state=True)
    
    return multi_rnn


### EXTRACT STATE OUTPUT OF MULTICELL-RNNS
def create_concat_state(state, num_layers, RNN_type):
    '''
        GOAL	     : concatenate the tuple-type tensor (state) into a single tensor
        state        : input state is a tuple ofo MulticellRNN (i.e. output of MulticellRNN)
                       consist of only hidden states h for GRU and hidden states c and h for LSTM
        num_layers   : number of layers in MulticellRNN
        RNN_type     : either 'LSTM' or 'GRU'
    '''
    for i in range(num_layers):
        if RNN_type == 'LSTM':
            tmp = state[i][1] ## i-th layer, h state for LSTM
        elif RNN_type == 'GRU':
            tmp = state[i] ## i-th layer, h state for GRU
        else:
            print('ERROR: WRONG RNN CELL TYPE')

        if i == 0:
            rnn_state_out = tmp
        else:
            rnn_state_out = tf.concat([rnn_state_out, tmp], axis = 1)
    
    return rnn_state_out


### FEEDFORWARD NETWORK

def create_FCNet(inputs, num_layers, h_dim, h_fn, o_dim, o_fn, w_init=None, keep_prob=1.0, w_reg=None):
    '''
        GOAL             : Create FC network with different specifications 
        inputs (tensor)  : input tensor
        num_layers       : number of layers in FCNet
        h_dim  (int)     : number of hidden units
        h_fn             : activation function for hidden layers (default: tf.nn.relu)
        o_dim  (int)     : number of output units
        o_fn             : activation function for output layers (default: None)
        w_init           : initialization for weight matrix (default: Xavier (GlorotUniform))
        keep_prob        : keep probability [0, 1]  (if 1.0, dropout is not employed)
    '''

    # Set default activation functions (hidden: relu, out: None)
    if h_fn is None:
        h_fn = 'relu'
    if o_fn is None:
        o_fn = None

    # Set default weight initialization (GlorotUniform in TF2.x)
    if w_init is None:
        w_init = GlorotUniform()

    # Construct the fully connected network
    for layer in range(num_layers):
        if num_layers == 1:
            # Single layer network
            out = Dense(o_dim, activation=o_fn, kernel_initializer=w_init, kernel_regularizer=w_reg)(inputs)
        
        else:
            if layer == 0:
                # First layer
                h = Dense(h_dim, activation=h_fn, kernel_initializer=w_init, kernel_regularizer=w_reg)(inputs)
                if keep_prob < 1.0:  # Only apply dropout if keep_prob is less than 1
                    h = Dropout(1 - keep_prob)(h)

            elif layer > 0 and layer != (num_layers - 1):
                # Intermediate layers
                h = Dense(h_dim, activation=h_fn, kernel_initializer=w_init, kernel_regularizer=w_reg)(h)
                if keep_prob < 1.0:
                    h = Dropout(1 - keep_prob)(h)

            else:
                # Last layer
                out = Dense(o_dim, activation=o_fn, kernel_initializer=w_init, kernel_regularizer=w_reg)(h)

    return out