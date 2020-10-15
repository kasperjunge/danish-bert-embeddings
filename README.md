# Danish BERT Embeddings
Package for making danish word+sentence embeddings with BERT 📜

## How to install?
```
pip install danish-bert-embeddings
```

## How to use it?
```
from danish_bert_embeddings import BertEmbeddingsDK
bert = BertEmbeddingsDK()
embedding = bert.embed('sætning eller ord som du gerne vil have embedded')
```

### Thanks to
- Lars Kjeldgaard (GitHub: smaakage85) for helping me package the project and publish it on PyPi.
_ Jens Dahl Møllerhøj (GitHub: mollerhoj) og BotXo for pre-training a danish version of BERT.



