'''
This declare DeepHit architecture:

INPUTS:
    - input_dims: dictionary of dimension information
        > x_dim: dimension of features
        > num_Event: number of competing events (this does not include censoring label)
        > num_Category: dimension of time horizon of interest, i.e., |T| where T = {0, 1, ..., T_max-1}
                      : this is equivalent to the output dimension
    - network_settings:
        > h_dim_shared & num_layers_shared: number of nodes and number of fully-connected layers for the shared subnetwork
        > h_dim_CS & num_layers_CS: number of nodes and number of fully-connected layers for the cause-specific subnetworks
        > active_fn: 'relu', 'elu', 'tanh'
        > initial_W: Xavier initialization is used as a baseline

LOSS FUNCTIONS:
    - 1. loglikelihood (this includes log-likelihood of subjects who are censored)
    - 2. rankding loss (this is calculated only for acceptable pairs; see the paper for the definition)
    - 3. calibration loss (this is to reduce the calibration loss; this is not included in the paper version)
'''

import numpy as np
import tensorflow as tf
import random

#from tensorflow.keras.layers import Dense as FC_Net
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import legacy as optimizers
#Faster for M1/M2 macs -N
### user-defined functions
import utils_network as utils

_EPSILON = 1e-08



##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + _EPSILON)

def div(x, y):
    return tf.div(x, (y + _EPSILON))

import tensorflow as tf

class Model_DeepHit(tf.keras.Model):
    def __init__(self, name, input_dims, network_settings):
        super(Model_DeepHit, self).__init__(name=name)

        # INPUT DIMENSIONS
        self.x_dim = input_dims['x_dim']
        self.num_Event = input_dims['num_Event']
        self.num_Category = input_dims['num_Category']

        # NETWORK HYPER-PARAMETERS
        self.h_dim_shared = network_settings['h_dim_shared']
        self.h_dim_CS = network_settings['h_dim_CS']
        self.num_layers_shared = network_settings['num_layers_shared']
        self.num_layers_CS = network_settings['num_layers_CS']

        self.active_fn = network_settings['active_fn']
        self.initial_W = network_settings['initial_W']

        self.reg_W = tf.keras.regularizers.l2(1e-4)
        self.reg_W_out = tf.keras.regularizers.l1(1e-4)

        # Build the network layers
        self.shared_layers = self.build_shared_layers()
        self.cause_specific_layers = self.build_cause_specific_layers()
        self.output_layer = self.build_output_layer()

    def build_shared_layers(self):
        layers = []
        for _ in range(self.num_layers_shared):
            layers.append(
                tf.keras.layers.Dense(self.h_dim_shared, activation=self.active_fn, kernel_initializer=self.initial_W,
                                      kernel_regularizer=self.reg_W)
            )
        return layers

    def build_cause_specific_layers(self):
        layers = []
        for _ in range(self.num_Event):
            event_layers = []
            for _ in range(self.num_layers_CS):
                event_layers.append(
                    tf.keras.layers.Dense(self.h_dim_CS, activation=self.active_fn, kernel_initializer=self.initial_W,
                                          kernel_regularizer=self.reg_W)
                )
            layers.append(event_layers)
        return layers

    def build_output_layer(self):
        return tf.keras.layers.Dense(self.num_Event * self.num_Category, activation='softmax',
                                     kernel_initializer=self.initial_W, kernel_regularizer=self.reg_W_out)

    def call(self, inputs, training=False):
        # Forward pass through shared layers
        x = inputs
        for layer in self.shared_layers:
            x = layer(x)

        # Forward pass through cause-specific layers
        outputs = []
        for event_layers in self.cause_specific_layers:
            h = x
            for layer in event_layers:
                h = layer(h)
            outputs.append(h)

        # Stack outputs for each event and reshape
        out = tf.stack(outputs, axis=1)
        out = tf.reshape(out, [-1, self.num_Event * self.h_dim_CS])

        # Dropout layer (the Dropout layer will handle keep_prob based on whether it's training or not)
        out = tf.keras.layers.Dropout(0.4)(out, training=training)  # Example keep_prob = 0.6, so Dropout rate = 1 - keep_prob

        # Final output layer
        out = self.output_layer(out)
        out = tf.reshape(out, [-1, self.num_Event, self.num_Category])

        return out

    def compute_loss(self, DATA, MASK, PARAMETERS, predictions):
        (x_mb, k_mb, t_mb) = DATA
        (m1_mb, m2_mb) = MASK
        (alpha, beta, gamma) = PARAMETERS

        I_1 = tf.sign(k_mb)

        # Compute log-likelihood loss (Loss 1)
        tmp1 = tf.reduce_sum(tf.reduce_sum(m1_mb * predictions, axis=2), axis=1, keepdims=True)
        tmp1 = I_1 * tf.math.log(tmp1)

        tmp2 = tf.reduce_sum(tf.reduce_sum(m1_mb * predictions, axis=2), axis=1, keepdims=True)
        tmp2 = (1.0 - I_1) * tf.math.log(tmp2)

        LOSS_1 = -tf.reduce_mean(tmp1 + 1.0 * tmp2)

        # Return the computed loss (you can include ranking and calibration loss as needed)
        return alpha * LOSS_1  # Add the other losses like ranking and calibration here
    
    def train_step(self, DATA, MASK, PARAMETERS, keep_prob, lr_train):
        with tf.GradientTape() as tape:
            # Forward pass through the model
            predictions = self(DATA[0], training=True)
            
            # Compute loss based on model predictions
            loss = self.compute_loss(DATA, MASK, PARAMETERS, predictions)

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_train)
        optimizer = optimizers.Adam(learning_rate=lr_train)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def predict(self, x_test, keep_prob=1.0):
        return self.call(x_test, training=False)