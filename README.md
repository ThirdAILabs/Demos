<div id="top"></div>

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/ThirdAILabs/Demos">
    <img src="https://www.thirdai.com/wp-content/uploads/2022/06/ThirdAI_logo.png" alt="Logo" width="150" height="">
  </a>

<h1 align="center">Demos</h1>

  <p align="center">
    Interactive notebooks for exploring the ThirdAI library.
    <br>
    <br>
    <a href="https://thirdai.com">[Website]</a>
    ·
    <a href="https://github.com/ThirdAILabs/Demos/issues">[Report Issues]</a>
    ·
    <a href="https://www.thirdai.com/careers/">[Careers]</a>
  </p>
</div>


<!-- ABOUT THE PROJECT -->
# 👋 Welcome

All of ThirdAI's technology is powered by its BOLT library. BOLT is a deep-learning framework that leverages sparsity to enable training and deploying very large scale deep learning models on any CPU. This demos repo will help get you familiar with our products [Neural DB](https://medium.com/thirdai-blog/thirdais-private-and-personalizable-neural-database-enhancing-retrieval-augmented-generation-f3ad52c54952) and [Universal Deep Transformer (UDT)](https://www.thirdai.com/universal-deep-transformers/) through interactive notebooks.

# 🧠 NeuralDB (for RAG and Search)

NeuralDB is an efficient, private, teachable CPU-only text retrieval engine. You can insert all your PDFs, DOCXs, CSVs (and even parse URLs) into a NeuralDB and do semantic search and QnA on them. Read our three part blog on why you need NeuralDB [here](https://medium.com/thirdai-blog/understanding-the-fundamental-limitations-of-vector-based-retrieval-for-building-llm-powered-48bb7b5a57b3). Leveraging over a decade of research in efficient neural network training, NeuralDB has been meticulously optimized to operate effectively on conventional CPUs, making it accessible to any standard desktop machine. Additionally, since it can be trained and used anywhere, NeuralDB gives you airgapped privacy, ensuring your data never leaves your local machine. 

With the capacity to scale Retreival Augmented Generation (RAG) capabilities over thousands of pages, NeuralDB revolutionizes the way you interact with your data.

Here is a quick overview of how NeuralDB works:

```python
from thirdai import neural_db as ndb

db = neural_db.NeuralDB()

db.insert(
  sources=[ndb.PDF(filename), ndb.DOCX(filename), ndb.CSV(filename)], 
  train=True
)

results = ndb.search(
    query="what is the termination period of this contract?",
    top_k=2,
)

for result in results:
    print(result.text)

```

NeuralDB also provides teaching methods for incorporating human feedback into RAG.

```python
# associate a source with a target
db.associate(source="parties involved", target="made by and between")

# associate text with a result
db.text_to_result("made by and between",0)
```

See the `neural_db` folder for more examples and documentation. 

# 🪐 Universal Deep Transformer (for all Transformer and ML needs)

Universal Deep Transformer (UDT) is our consolidated API for performing different ML tasks on a variety of data types. It handles text, numeric, categorical, multi-categorical, graph, and time series data while generalizing to tasks like NLP, multi-class classification, multi-label retrieval, regression etc. Just like NeuralDB, UDT is optimized for conventional CPUs and is accessible to any standard desktop machine. 

Some applications of UDT include:
* [Text Classification](https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer/text%20classification)
* [Named Entity Recognition](https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer/named_entity_recognition)
* [Tabular Data Classification](https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer/tabular_classification)
* [Netflix-style Movie Recommendation](https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer/personalization_and_recommendation)
* [Query Reformulation](https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/QueryReformulation.ipynb)
* [Graph Node Classification](https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/graph_neural_networks/GraphNodeClassification.ipynb)
* [Sentiment Analysis](https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/text%20classification/SentimentAnalysis.ipynb)
* [Intent Classification](https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/text%20classification/IntentClassification.ipynb)
* [Zero Shot Search and Retrieval](https://github.com/ThirdAILabs/Demos/tree/main/universal_deep_transformer/llm_search) 
* [Fraud Detection](https://github.com/ThirdAILabs/Demos/blob/main/universal_deep_transformer/tabular_classification/FraudDetection.ipynb)
* and more!

Here is an example of the UDT API used for multi-label tabular classification:

```python
from thirdai import bolt

model = bolt.UniversalDeepTransformer(
    data_types={
        "title": bolt.types.text(),
        "category": bolt.types.categorical(),
        "number": bolt.types.numerical(range=(0, 100)),
        "label": bolt.types.categorical(delimiter=":")
    },
    target="label",
    n_target_classes=2,
    delimiter='\t',
)

model.train(filename.csv, epochs=5, learning_rate=0.001, metrics=["precision@1"])

model.predict({"title": "Red shoes", "category": "XL", "number": "12.6"})
```

See the `universal_deep_transformer` folder for more examples and documentation. 

<!-- LICENSE -->
# 📄 License

Many notebooks come with an API key that will only work on the dataset in the demo. If you want to try out ThirdAI on your own dataset, simply register for a free license [here](https://www.thirdai.com/try-bolt/).

To use your license do the following before constructing your NeuralDB or UDT models.
```python
from thirdai import licensing

licensing.activate("") # insert your valid license key here

# create NeuralDB or UDT ...
```

Please refer to `LICENSE.txt` for more information on usage terms.

<!-- CONTACT -->
# 🎙 Contact

ThirdAILabs - [@ThirdAILab](https://twitter.com/ThirdAILab) - [contact@thirdai.com](mailto:contact@thirdai.com)


<!-- MARKDOWN LINKS & IMAGES -->
[forks-shield]: https://img.shields.io/github/forks/thirdailabs/demos.svg?style=for-the-badge
[forks-url]: https://github.com/ThirdAILabs/Demos/network/members
[stars-shield]: https://img.shields.io/github/stars/thirdailabs/demos.svg?style=for-the-badge
[stars-url]: https://github.com/ThirdAILabs/Demos/stargazers
[issues-shield]: https://img.shields.io/github/issues/thirdailabs/demos.svg?style=for-the-badge
[issues-url]: https://github.com/ThirdAILabs/Demos/issues
[license-shield]: https://img.shields.io/github/license/thirdailabs/demos.svg?style=for-the-badge
[license-url]: https://github.com/ThirdAILabs/Demos/blob/master/LICENSE.txt
