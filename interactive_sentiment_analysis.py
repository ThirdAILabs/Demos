from thirdai import bolt
import numpy as np
import mmh3

def demo(network: bolt.Network):
    include_bigrams = True
    k = 0
    seed = 0
    feat_hash_dim = 100000
    while True:
        inp = input('Input:')
        line = inp.lower().strip()
        itms = line.split(' ')
        ## we can add stemming, lower casing etc here
        x_idxs = [mmh3.hash(itm, seed) % feat_hash_dim for itm in itms]
        if include_bigrams:
            x_idxs += [mmh3.hash(itms[j]+'-'+itms[j+1], seed)%feat_hash_dim for j in range(len(itms)-2)]
        if k>0:
            x_idxs += [mmh3.hash(line[j:j+k], seed)%feat_hash_dim for j in range(len(line)+1-k)]
        ##
        x_offsets = np.int32([0,len(x_idxs)])
        label = np.array([np.uint32(0)])
        x_vals = np.ones(len(x_idxs))
        ######
        temp = network.predict(x_idxs, x_vals, x_offsets, label, [0,1], batch_size=1)
        if temp>0:
            print('positive!')
        else:
            print('negative!')