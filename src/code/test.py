if __name__=='__main__':
    import numpy as np
    obs = np.asarray([1,2,3,5,-100,-4])

    a = np.asarray([1,2,3])
    b = np.asarray([3,4,5])
    v = np.outer(a,b)
    print(v)
    exit(0)
    wts = ['wi_h1','wh1_h2','wh2_o']
    bias = ['ao_inp','ao_h1','ao_h2','ao_op']
    l_c = ['l2','l3','l4']

    # print(np.outer(a.flatten(),b.flatten()))
    for i in reversed(range(1,len(b)-1)):
        print(wts[i],' : ',bias[i],' : ',bias[i-1])
        # print(i)
    exit(0)
    # print(a)

    from tqdm import tqdm
    # import split
    epochs = 2
    a = [1,2,3,4]
    b = ['a','b','c','d']
    res = int(100/32)
    print(res)
    b = [46,38,94,25,57,92,98,9,23,91,6,68,48,1,70,84,5,2,0,52,85,87,78,51,45,60,47,32,11,79,74,42,36,75,4,73,44,54,71,34,41,90,63,77,53,86,49,99,69,76,83,43,33,39,88,56,61,80,40,12,59,89,31,17,14,30,72,16,24,62,65,95,3,55,22,81,21,66,82,15,50,96,19,26,97,13,58,20,67,10,18,37,27,28,8,93]

    print(a)
    # exit(0)
    # for i in a:
    #     print(i)
    a = np.array_split(b, 5)
    for i,elem in enumerate(a):
        print('############################### ')
        c= []
        b = np.concatenate(a[:i]+a[i+1:])
        print(b)
    exit(0)
    # exit(0)
    for epoch in tqdm(range(epochs)):
        print('############################################')
        a = np.random.choice(range(100),(res,32),replace=False)
        for i in a:
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print(a)
