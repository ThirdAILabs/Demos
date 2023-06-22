import thirdai
from thirdai import bolt, licensing

try:
    licensing.activate("D0F869-B61466-6A28F0-14B8C6-0AC6C6-V3")
except:
    pass

import pickle

import numpy as np
from thirdai import hashing

########### Extract params ###########
model = bolt.UniversalDeepTransformer.load("hotpotqa_to_msmarco.bolt")

# W1 = model._get_model().get_layer('fc_1').weights.get()
# b1 = model._get_model().get_layer('fc_1').biases.get()
# W2 = model._get_model().get_layer('fc_2').weights.get()
# b2 = model._get_model().get_layer('fc_2').biases.get()

W1 = model._get_model().ops()[0].weights
b1 = model._get_model().ops()[0].biases
ht1 = model._get_model().ops()[0].get_hash_table()
hash_fn_1 = ht1[0]
hash_table_1 = ht1[1]
ht1[0].save("hash_fn_1")
ht1[1].save("hash_table_1")


W2 = model._get_model().ops()[1].weights
b2 = model._get_model().ops()[1].biases
ht2 = model._get_model().ops()[1].get_hash_table()
hash_fn_2 = ht2[0]
hash_table_2 = ht2[1]
ht2[0].save("hash_fn_2")
ht2[1].save("hash_table_2")

np.save("W1.npy", W1)
np.save("b1.npy", b1)
np.save("W2.npy", W2)
np.save("b2.npy", b2)

import pickle

import numpy as np
import thirdai

############ Set Params #################
from thirdai import bolt, dataset, hashing

# tokenizer = dataset.WordpieceTokenizer("vocab")

model = bolt.UniversalDeepTransformer(
    data_types={
        "QUERY": bolt.types.text(),
        "DOC_ID": bolt.types.categorical(delimiter=":"),
    },
    target="DOC_ID",
    n_target_classes=835311,
    integer_target=True,
    options={
        # "extreme_classification":True,
        # "extreme_output_dim":50000,
        # "extreme_num_hashes": 8,
        # "input_dim":50000,
        "embedding_dimension": 1024,
        "use_tanh": True,
        "use_bias": True,
    },
)

W1 = np.load("W1.npy")
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

hash_fn_1 = hashing.DWTA.load("hash_fn_1")
hash_table_1 = bolt.nn.HashTable.load("hash_table_1")

hash_fn_2 = hashing.DWTA.load("hash_fn_2")
hash_table_2 = bolt.nn.HashTable.load("hash_table_2")

model._get_model().ops()[0].set_weights(W1)
model._get_model().ops()[0].set_biases(b1)
model._get_model().ops()[1].set_weights(W2)
model._get_model().ops()[1].set_biases(b2)

model._get_model().ops()[0].set_hash_table(hash_fn_1, hash_table_1)
model._get_model().ops()[1].set_hash_table(hash_fn_2, hash_table_2)

model.save("pubmed_800k_0.7.6.bolt")
