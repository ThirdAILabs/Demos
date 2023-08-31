# NeuralDB

## Introduction

Welcome to NeuralDB, an efficient, private, teachable database for neural text search over your documents. Leveraging over a decade of research in efficient neural network training, NeuralDB has been meticulously optimized to operate effectively on conventional CPUs, making it accessible to any standard desktop machine. Additionally, since it can be trained and used anywhere, NeuralDB gives you airgapped privacy, ensuring your data never leaves your local machine. 

With the capacity to scale search capabilities over thousands of pages, NeuralDB revolutionizes the way you interact with your data.

## Creating your NeuralDB

This step will assume you have installed the thirdai python package and obtained a valid license. Our python package is available through pip via ```pip3 install thirdai```. To obtain a license please visit https://www.thirdai.com/try-bolt 

```python
from thirdai import neural_db, licensing

licensing.activate("YOUR LICENSE KEY HERE")

ndb = neural_db.NeuralDB()
```

Additionally, we provide a suite of pre-trained databases from our Model Bazaar. To download a pre-trained model:

```python
from thirdai.model_bazaar import Bazaar

bazaar = Bazaar()
bazaar.fetch()

# This will return all the available pre-trained models
print(bazaar.list_model_names())

# pass the string identifier of the model you'd like to use. General QnA is our 
# most generic and foundational pre-trained model for the majority of use cases. 
ndb = bazaar.get_model("General QnA")
```

## Documents

NeuralDB uses your documents to create an intelligent search engine. It supports a wide array of document formats including CSV, PDF, and DOCX.

```python
pdf_doc = neural_db.PDF(filename)

docx_doc = neural_db.DOCX(filename)

csv_doc = neural_db.CSV(
        filename,
        id_column="id", # expected to be between [0 and num_rows)
        strong_columns=["title"], # a "strong" text signal like titles or tags
        weak_columns=["text"], # a "weak" text signal like description or bullets
        reference_columns=["id", "title", "text"], # columns you'd like shown in subsequent search results
)
```

## Inserting and Searching Documents

To insert your documents into NeuralDB, use the insert() function. This function accepts a list of documents and a train flag:

```python
ndb.insert(sources=[doc, pdf_doc, docx_doc], train=True)
```

You can search your documents using the search() function:

```python
results = ndb.search(
    query="what is the termination period of this contract?",
    top_k=2,
)

for result in results:
    print(result.text)
```

## Teaching

NeuralDB improves with human feedback. You can provide feedback using two APIs: associate() and text_to_result().

```python
# associate a source with a target
db.associate(source="parties involved", target="made by and between")

# associate text with a result
db.text_to_result("made by and between",0)
```
These functions enable the NeuralDB system to learn and improve its search capabilities over time.

## FAQ

**Q: How many documents can NeuralDB handle?**

A: NeuralDB is designed to be scalable and can handle thousands of pages efficiently.

**Q: Which types of documents are supported by NeuralDB?**

A: NeuralDB currently supports CSV, PDF, and DOCX formats. Additionally ask us about our URL document which includes built-in web scraping features.

**Q: What are some best practices around retraining?**

A: If you have to insert new documents greater than about 10% the size of your current data size, we recommend retraining the database. On frequent incremental additions this will ensure that search results are balanced across your data. To retrain the database, do the following
```python
db.clear_sources() # forgets all the documents
db.insert(documents, train=True)
```

Additionally, we always recommend using train=True when possible to maximize the learning capability that NeuralDB offers. 


## Contact Support

For further questions or issues, please contact our support team at contact@thirdai.com