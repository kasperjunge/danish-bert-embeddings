# Danish BERT Embeddings
<img align='right' src="https://user-images.githubusercontent.com/39537120/96163240-f1c8cd80-0f19-11eb-8bb5-ab1e9f467060.jpg" width="500"><br>
Tired of not having any Danish BERT embeddings? Well, you came to the right place! <br><br>
This is a package for making danish word/sentence embeddings for your very own NLP project, with BERT! 
See how to get started below ⬇️<br><br>

## How to install?
```
pip install danish-bert-embeddings
```
## How to use it?
```
from danish_bert_embeddings import DanishBertEmbeddings

embedder = DanishBertEmbeddings()
embedding = embedder.embed('En mega fed embedding.')
```
### Thanks to
- Lars Kjeldgaard (GitHub: smaakage85) for helping me packaging the project and publish it on PyPi.
- Jens Dahl Møllerhøj (GitHub: mollerhoj) and BotXo for pre-training a danish version of BERT.
- Malte Højmark-Bertelsen for uploading danish bert from BotXo to Huggingface https://huggingface.co/Maltehb/danish-bert-botxo
<br>



