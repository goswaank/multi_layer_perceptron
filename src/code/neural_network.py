import numpy as np
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
class Neural_Network:

    def __init__(self, layer_dim, activation):
        self.layers = layer_dim
        self.num_of_layers = len(layer_dim)
        self.activation = activation
        self.weight_store = []
        self.bias_store = []
        self.activation_store = []
        self.accuracy = []
        self.error = []
        self.model = []
        self.layer_details = []
        self.output_labels_range = layer_dim[-1]

    def weight_initializer(self, input_dim, output_dim, initialization_type):
        if initialization_type=='uniform':
            weights = np.random.uniform(-1/math.sqrt(input_dim), 1/math.sqrt(input_dim),[output_dim,input_dim])
        elif initialization_type == 'normal':
            weights = np.random.normal(0,0.5,[output_dim,input_dim])
        return weights

    def bias_initializer(self,layer_len, initialization_type):
        if initialization_type == 'uniform':
            bias = np.reshape(np.random.uniform(-1,1,[layer_len]),(layer_len,1))
        if initialization_type == 'normal':
            bias = np.reshape(np.random.normal(0,0.5,[layer_len]),(layer_len,1))
        return bias


    def get_activation(self,vector,activation_type):
        if activation_type == 'sigmoid':
            ## formula => f(x) = 1/(1+e^(-x))
            activation = 1./(1.+np.exp(-vector))
            return np.round(activation,decimals=3)

        elif activation_type == 'relu':
            ## Formula => x if x>=0 and 0 otherwise
            activation = [x if x >= 0 else 0 for x in np.ndarray.flatten(vector)]
            activation = np.reshape(activation,(len(activation),1))
            return np.round(activation,decimals=3)

        elif activation_type == 'softmax':
            ## Formula => (exp(x))/Sum(exp(k) for all k)
            vector = vector - np.max(vector)
            vector_exp = np.exp(vector)
            vector_sum = np.sum(vector)
            activation = vector_exp/vector_sum
            activation = np.reshape(activation, (len(activation), 1))
            return np.round(activation,decimals=3)

        elif activation_type == 'tanh':
            ## Formula => (exp(x)-exp(-x))/(exp(x)+exp(-x))
            vector_pos_exp = np.exp(vector)
            vector_neg_exp = np.exp(-vector)
            activation = np.divide((vector_pos_exp-vector_neg_exp),(vector_pos_exp+vector_neg_exp))
            activation = np.reshape(activation, (len(activation), 1))
            return np.round(activation,decimals=3)

    def get_gradient(self,activ_input, activation_type):
        if activation_type == 'sigmoid':
            ## formula => f(x)(1-f(x))
            gradient = activ_input - np.multiply(activ_input,activ_input)
            return gradient

        elif activation_type == 'relu':
            ## Formula => 1 if x>=0 and 0 otherwise
            gradient = [1 if x >= 0 else 0 for x in np.ndarray.flatten(activ_input)]
            return np.reshape(gradient,(len(gradient),1))

        elif activation_type == 'softmax':
            ## Formula => f(x)(1-f(x))
            gradient = activ_input * np.multiply(activ_input,activ_input)
            return gradient

        elif activation_type == 'tanh':
            ## Formula => 1 - (f(x))^2
            gradient = 1 - np.multiply(activ_input, activ_input)
            return gradient


    def build_model(self):
        for i,layer in enumerate(self.layers[1:]):
            input_dim = self.layers[i]
            output_dim = self.layers[i+1]
            activation = self.activation[i]
            layer_i = {'weights':self.weight_initializer(input_dim,output_dim,'uniform'),'bias' : self.bias_initializer(output_dim,'uniform'),'activation':activation}
            self.model.append(layer_i)

    def forward_prop(self,obs):
        inp = obs
        self.layer_details = []
        self.layer_details.append({'data':obs})
        for i in range(self.num_of_layers-1):
            # print('self.model[i][weights] : ',type(self.model[i]['weights']), ' : ',self.model[i]['weights'],' : ',type(self.model[i]['weights'][0]),' : ',type(self.model[i]['weights'][0][0]))
            # print('inp : ',type(inp),' : ',inp,' : ',type(inp[0]))
            # print('wts : ',np.shape(self.model[i]['weights']))
            # print('inp : ',np.shape(inp))
            weighted_inp = np.matmul(self.model[i]['weights'],inp) + self.model[i]['bias']
            # print(self.model[i]['activation'])
            activated_op = self.get_activation(weighted_inp,self.model[i]['activation'])
            # print('activ_op : ', np.shape(activated_op))
            inp = activated_op


            layer_contrib = {'data':activated_op}
            # print(np.shape(activated_op))
            self.layer_details.append(layer_contrib)
        return activated_op

    def backward_prop(self,sensitivity,output,eta):

        gradient = self.get_gradient(output,self.model[-1]['activation'])
        delta = np.multiply(sensitivity,gradient)
        new_weights = []
        new_bias = []
        new_bias.insert(0,self.model[-1]['bias'] - np.multiply(eta,delta))
        new_weights.insert(0,self.model[-1]['weights'] - np.multiply(eta,np.outer(delta, self.layer_details[-2]['data'])))

        # new_delta = np.matmul(delta,np.transpose(self.weight_store[-1]))
        for i in reversed(range(1,self.num_of_layers-1)):
            sensitivity = np.matmul(np.transpose(self.model[i]['weights']), delta)
            gradient = self.get_gradient(self.layer_details[i]['data'],self.model[i]['activation'])
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%5')
            # print(np.shape(sensitivity))
            # print(np.shape(gradient))
            # print(self.model[i]['activation'])
            delta = np.multiply(sensitivity,gradient)
            new_bias.insert(0,self.model[i-1]['bias']-np.multiply(eta,delta))
            # print('##################################')
            # print(np.shape(self.model[i-1]['weights']))
            # print(np.shape(delta))
            # print(np.shape(self.layer_details[i-1]['data']))
            # print(np.shape(np.multiply(eta,np.outer(delta,self.layer_details[i-1]['data']))))
            new_weights.insert(0,np.subtract(self.model[i-1]['weights'],np.multiply(eta,np.outer(delta,self.layer_details[i-1]['data']))))
        for i in range(len(new_weights)):
            self.model[i]['weights'] = new_weights[i]
            self.model[i]['bias'] = new_bias[i]

    def predict_binary_score(self, target, output):
        op_arg = np.argmax(output)
        score = 1 if op_arg == np.argmax(target) else 0
        return score

    def train(self,records,labels,epochs,batch_size,eta):
        num_of_obs = len(records)
        input_size = len(records[1])
        train_error = []
        splits = np.array_split(range(num_of_obs), 5)
        train_ind = np.concatenate(splits[:-1])
        test_ind = splits[-1]
        train_size = len(train_ind)
        # print('train_size : ',train_size)
        # print('batch_size : ',batch_size)
        # exit(0)
        num_of_samples = int(train_size/batch_size)

        for epoch in tqdm(range(epochs)):
            samples = np.random.choice(range(train_size), (num_of_samples, batch_size), replace=False)
            # print(samples)
            acc_sample = 0
            acc = 0
            err = 0

            for sample in samples:
                error_sample = 0
                acc_sample = 0
                for ind in sample:
                    obs = np.reshape(records[ind],(input_size,1))
                    output = self.forward_prop(obs)
                    target = labels[ind]
                    op_tgt_diff = np.subtract(output,target)
                    acc = acc + self.predict_binary_score(target=target, output=output)

                    err = err + 0.5*(np.matmul(np.transpose(op_tgt_diff),op_tgt_diff))
                # print('_____________________****_________ ',acc)
                self.backward_prop(np.divide(op_tgt_diff, batch_size),output,eta)

                error_sample = error_sample + err/batch_size
                acc_sample = acc_sample + acc/batch_size
            # print('_________________________',acc_sample)
            train_accuracy = acc_sample/num_of_samples
            # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%% ',train_accuracy)
            train_error = error_sample/num_of_samples

            acc = 0
            err = 0
            for ind in test_ind:
                obs = np.reshape(records[ind], (input_size, 1))
                output = self.forward_prop(obs)
                target = labels[ind]
                op_tgt_diff = np.subtract(output,target)
                err = err + 0.5 * (np.matmul(np.transpose(op_tgt_diff), op_tgt_diff))
                # print('--------------- ',err)
                acc = acc + self.predict_binary_score(target=target, output=output)
            test_accuracy = acc/len(test_ind)
            test_error = err/len(test_ind)
            # print('\n================== ',test_error)
            self.accuracy.append({'train':train_accuracy,'test':test_accuracy})
            # print('\n********************* ',{'train':train_error,'test':test_error})
            self.error.append({'train':train_error,'test':test_error})
        self.model
        return self.model

    def plot_loss(self, inp_title):
        train_loss = [elem['train'] for elem in self.error]
        test_loss = [elem['test'] for elem in self.error]

        plt.figure(figsize=(10,8))
        train_loss = np.concatenate(train_loss)
        test_loss = np.concatenate(test_loss)

        plt.plot(train_loss, label='Train')
        plt.plot(test_loss, label='Test')
        plt.title(inp_title)
        plt.legend()
        plt.grid(True)
        plt.xlabel("loss curve")
        outpath = os.path.join(os.getcwd() +
                               '/plots/loss.png')
        plt.savefig(outpath)

    def plot_accuracy(self, inp_title):
        train_accuracy = [elem['train'] for elem in self.accuracy]
        test_accuracy = [elem['test'] for elem in self.accuracy]

        plt.figure(figsize=(10,8))
        plt.plot(train_accuracy, label='Train')
        plt.plot(test_accuracy, label='Test')
        plt.title(inp_title)
        plt.legend()
        plt.grid(True)
        plt.xlabel("accuracy curve")
        outpath = os.path.join(os.getcwd() +
                               '/plots/accuracy.png')
        plt.savefig(outpath)
