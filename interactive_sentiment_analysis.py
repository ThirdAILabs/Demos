from thirdai import bolt
import numpy as np
import mmh3

def demo(network: bolt.Network, verbose: bool):
    network.enable_sparse_inference()
    include_bigrams = True
    k = 0
    seed = 0
    feat_hash_dim = 100000
    
    while True:
        inp = input('Input:')
        line = inp.lower().strip()
        if line == "exit":
            print("Exiting demo...")
            break
        itms = line.split(' ')
        
        ## we can add stemming, lower casing etc here
        x_idxs = [mmh3.hash(itm, seed) % feat_hash_dim for itm in itms]
        if include_bigrams:
            x_idxs += [mmh3.hash(itms[j]+'-'+itms[j+1], seed)%feat_hash_dim for j in range(len(itms)-2)]
        if k>0:
            x_idxs += [mmh3.hash(line[j:j+k], seed)%feat_hash_dim for j in range(len(line)+1-k)]
        
        ## Format as bolt numpy input
        x_offsets = np.int32([0,len(x_idxs)])
        x_vals = np.ones(len(x_idxs))
        y_idxs = np.array([np.uint32(0)])
        y_vals = np.array([np.float32(1)])
        y_offsets = np.array([np.uint32(10), np.uint32(10)])
        
        ## Predict
        temp = network.predict(
            x_idxs, x_vals, x_offsets, 
            y_idxs, y_vals, y_offsets, 
            batch_size=1, 
            metrics=["categorical_accuracy"],
            verbose=verbose)
        
        pred = np.argmax(temp[1][0])
        
        if pred > 0:
            print('PREDICTION RESULT: POSITIVE', flush=True)
        else:
            print('PREDICTION RESULT: NEGATIVE', flush=True)

        print(f"Predicted in {temp[0]['test_time'][0]} milliseconds.\n")
