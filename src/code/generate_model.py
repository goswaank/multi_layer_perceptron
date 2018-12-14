import src.code.neural_network as nn
import pickle as pkl


if __name__=='__main__':
    import numpy as np
    with open('data/x_train.csv','r') as read_fp:
        x_data = read_fp.readlines()

    data = []
    for record in x_data:
        a = []
        for elem in record.strip('\n').split(','):
            a.append(int(elem))
        data.append(a)

    with open('data/y_train.csv','r') as read_fp:
        y_data = read_fp.readlines()


    target = []
    num_of_op_nodes = 10
    for record in y_data:
        binarized_record = np.zeros(num_of_op_nodes)
        binarized_record[int(record)] = 1
        target.append(np.reshape(binarized_record,(num_of_op_nodes,1)))


    network = nn.Neural_Network(layer_dim=[14, 100, 40, 10],activation=['relu','relu','softmax'])

    print("Building network computational graph..."),
    network.build_model(),
    print("Complete!")
    trained_model = network.train(records=data, labels=target, epochs=1000,
                       batch_size=32, eta=0.0005)

    print("Evaluating testing loss w.r.t epoches..."),
    network.plot_loss(inp_title='Network [14-100-40-10] Loss Curve'),
    print("Complete!")

    print("Evaluating training and testing accuracy w.r.t epoches..."),
    network.plot_accuracy(inp_title='Network [14-100-40-10] Accuracy Curve')
    print("Complete!")

    with open('model/model.pkl','wb') as fp:
        pkl.dump(trained_model,fp)
