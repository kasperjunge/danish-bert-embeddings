# Danish BERT Embeddings
<img align='right' src="https://user-images.githubusercontent.com/39537120/96163240-f1c8cd80-0f19-11eb-8bb5-ab1e9f467060.jpg" width="500"><br>
Tired of not having any Danish BERT embeddings? Well, you came to the right place, because this is a package for making danish word/sentence embeddings for your very own NLP project, with BERT! See how to get started below ⬇️

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
- Jens Dahl Møllerhøj (GitHub: mollerhoj) og BotXo for pre-training a danish version of BERT.



