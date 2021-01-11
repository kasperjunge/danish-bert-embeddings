import torch
from transformers import AutoTokenizer, AutoModelForPreTraining



class DanishBertEmbeddings():
    """Class for creating Danish BERT word and sentence on the fly. """

    
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
        self.model = AutoModelForPreTraining.from_pretrained("Maltehb/danish-bert-botxo")

    def embed(self, sentence, output_numpy=False):
        """
        Parameters
        ----------
        sentence : str
            Sentence to be embedded.
        output_numpy : bool
            Whether to return numpy or not. Default: False
        
        Returns
        -------
        embedding : torch.tensor / numpy.array
            Sentence embedding.
        """

        # tokenize
        tokenized = self.tokenizer(sentence, max_length=512, truncation=True, return_tensors='pt')
        
        # run through berty
        with torch.no_grad():
            output = self.model(**tokenized, output_hidden_states=True)

        # extract hidden staytes        
        hidden_states = torch.cat(output[2])

        # mean last layer to create sentence embedding
        embedding = torch.mean(hidden_states[-1:,:,:].squeeze(), dim=0)

        if output_numpy:
            embedding = embedding.numpy()
        
        return embedding
        
        