import pickle as pkl

if __name__=='__main__':
    with open('model/model.pkl','rb') as fp:
        res = pkl.load(fp)

    for elem in res:
        print('##################################')
        print(elem['weights'])
        print(elem['bias'])
        print(elem['activation'])
