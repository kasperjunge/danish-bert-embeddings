from danish_bert_embeddings import DanishBertEmbeddings

from danish_bert_embeddings.danish_bert_embeddings import DanishBertEmbeddings

embedder = DanishBertEmbeddings()
embedding = embedder.embed('hvad er penis')
print(embedding)