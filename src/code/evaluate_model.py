import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

class evaluator:
    model = []
    num_of_layers = 0

    def get_model(self):
        with open('model/model.pkl', 'rb') as fp:
            model = pkl.load(fp)
        weights = []
        bias = []
        for elem in model:
            weights.append(elem['weights'])
            bias.append(elem['bias'])
        return model

    def get_data(self):
        with open('data/x_test.csv', 'r') as fp:
            x_data = fp.readlines()
        data = []
        for record in x_data:
            local_list = []
            for item in record.strip('\n').split(','):
                local_list.append(int(item))
            data.append(local_list)
        return data

    def get_labels(self):
        with open('data/y_test.csv','r') as fp:
            y_data = fp.readlines()

        target = []
        num_of_op_nodes = 10
        for record in y_data:
            binarized_record = np.zeros(num_of_op_nodes)
            binarized_record[int(record)] = 1
            target.append(np.reshape(binarized_record, (num_of_op_nodes, 1)))
        return target

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


    def predict_binary_score(self, target, output):
        op_arg = np.argmax(output)
        score = 1 if op_arg == np.argmax(target) else 0
        return score

    def forward_prop(self, obs):
        inp = obs
        for i in range(self.num_of_layers - 1):
            weighted_inp = np.matmul(self.model[i]['weights'], inp) + self.model[i]['bias']
            activated_op = self.get_activation(weighted_inp, self.model[i]['activation'])
            inp = activated_op
        return activated_op

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
                               '/plots/test_loss.png')
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
                               '/plots/test_accuracy.png')
        plt.savefig(outpath)

    def evaluate_model(self):
        x_data = self.get_data()
        y_data = self.get_labels()
        self.model = self.get_model()
        self.num_of_layers = len(self.model)+1
        input_size = len(x_data[0])
        score = 0
        for ind,obs in enumerate(x_data):
            point = np.reshape(obs, (input_size, 1))
            output = self.forward_prop(point)
            target = y_data[ind]
            score = score + self.predict_binary_score(target=target,output=output)

        print(score/len(x_data))

if __name__=='__main__':
    e = evaluator()
    e.evaluate_model()