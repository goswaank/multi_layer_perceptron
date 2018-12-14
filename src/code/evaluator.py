import numpy as np
import math

class evaluate:

    def get_activation(self,vector,activation_type):
        if activation_type == 'sigmoid':
            ## formula => f(x) = 1/(1+e^(-x))
            activation = 1./(1.+np.exp(-vector))
            return np.round(activation,decimals=3)

        elif activation_type == 'relu':
            ## Formula => x if x>=0 and 0 otherwise
            activation = [x if x >= 0 else 0 for x in np.ndarray.flatten(vector)]
            return np.round(activation,decimals=3)

        elif activation_type == 'softmax':
            ## Formula => (exp(x))/Sum(exp(k) for all k)
            vector = vector - np.max(vector)
            vector_exp = np.exp(vector)
            vector_sum = np.sum(vector)
            activation = vector_exp/vector_sum
            return np.round(activation,decimals=3)

        elif activation_type == 'tanh':
            ## Formula => (exp(x)-exp(-x))/(exp(x)+exp(-x))
            vector_pos_exp = np.exp(vector)
            vector_neg_exp = np.exp(-vector)
            activation = np.divide((vector_pos_exp-vector_neg_exp),(vector_pos_exp+vector_neg_exp))
            return np.round(activation,decimals=3)

    def get_gradient(self,activ_input, activation_type):
        if activation_type == 'sigmoid':
            ## formula => f(x)(1-f(x))
            gradient = activ_input - np.multiply(activ_input,activ_input)
            return gradient

        elif activation_type == 'relu':
            ## Formula => 1 if x>=0 and 0 otherwise
            gradient = [1 if x >= 0 else 0 for x in np.ndarray.flatten(activ_input)]
            return gradient

        elif activation_type == 'softmax':
            ## Formula => f(x)(1-f(x))
            gradient = activ_input * np.multiply(activ_input,activ_input)
            return gradient

        elif activation_type == 'tanh':
            ## Formula => 1 - (f(x))^2
            gradient = 1 - np.multiply(activ_input, activ_input)
            return gradient


    def forward_prop(self,input_vector, weight_store, bias_store):
        for wt in weight_store:
            net = np.matmul(wt, input_vector)
