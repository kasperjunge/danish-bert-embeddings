import os
import requests
import torch
from transformers import BertConfig, BertTokenizer, BertModel

class BertEmbeddingsDK():
    """Class for creating Danish BERT word and sentence on the fly.
        
        Some more extensive description
        
        Attributes
        ----------
        bert_path : str 
            Path to danish bert model. 
        url : str
            Download url to BotXo's danish pre-trained BERT.
        config : transformers.configuration_bert.BertConfig
            Configuration object for bert.
        tokenizer : transformers.tokenization_bert.BertTokenizer
            Bert tokenizer.
        model : transformers.modeling_bert.BertModel
            Pre-trained danish bert model.
        """

    
    def __init__(self, bert_path=None):
        """
        Parameters
        ----------
        bert_path : str 
            Path to danish bert model. Default: ./bert-base-danish        
        """
        
        # Danish BERT url
        self.url = 'https://www.dropbox.com/s/19cjaoqvv2jicq9/danish_bert_uncased_v2.zip?dl=1'
        
        # If custom model path
        if bert_path == None: 
            self.bert_path = './bert-base-danish/'
        else: 
            self.bert_path = bert_path
            self._load_model()
    
        # Check model
        pt_check, tf_check = self._check_model_files()
        if pt_check:
            self._load_model()
        elif tf_check:
            self._convert_to_pytorch()
            self._load_model()
        else:
            self._download_danish_bert()
            self._convert_to_pytorch()
            self._load_model()
        
        
    
    def _check_model_files(self):
        print('Checking required model files..')
        tf_required_files = ['bert_model.ckpt.data-00000-of-00001', 'bert_model.ckpt.index',
                             'bert_model.ckpt.meta', 'bert_config.json', 'vocab.txt']
        pt_required_files = ['bert_model.ckpt.data-00000-of-00001', 'bert_model.ckpt.index',
                             'bert_model.ckpt.meta', 'config.json', 'pytorch_model.bin', 'vocab.txt']
        if os.path.isdir(self.bert_path):
            pt_check = all([file in pt_required_files for file in os.listdir(self.bert_path)])
            tf_check = all([file in tf_required_files for file in os.listdir(self.bert_path)])
        else:
            print('Model was not found..')
            pt_check = False
            tf_check = False

        return pt_check, tf_check
    
            
    def _load_model(self):
        print('Loading model..')
        self.config = BertConfig.from_json_file(self.bert_path+'config.json')
        self.tokenizer = BertTokenizer(vocab_file=self.bert_path+'vocab.txt')
        self.model = BertModel(config=self.config).from_pretrained(self.bert_path)
        self.model.eval()
        print('Ready for embedding!')
    
    def _convert_to_pytorch(self):
        print('Converting model to PyTorch..')
        os.system('transformers-cli convert --model_type bert --tf_checkpoint ./bert-base-danish/bert_model.ckpt --config ./bert-base-danish/bert_config.json --pytorch_dump_output ./bert-base-danish/pytorch_model.bin')
        os.system('mv bert-base-danish/bert_config.json bert-base-danish/config.json')
        return None
    
    def _download_danish_bert(self):
        print('Downloading danish BERT (this may take some time ^^)..')
        r = requests.get(self.url, allow_redirects=True)
        open('bert-base-danish.zip', 'wb').write(r.content)
        os.system('unzip bert-base-danish.zip')
        os.system('mv danish_bert_uncased_v2 bert-base-danish')
        os.system('rm bert-base-danish.zip')
        os.system('rm -rf __MACOSX')
        return None
    
    def _embedding_method(self):
        # bla bla sum + avg last x layers
        return None
    
    def embed(self, text, print_num_tokens=False):
        
        tokenized = self.tokenizer(text, return_tensors='pt')
        if print_num_tokens: print(tokenized['input_ids'].shape)
        with torch.no_grad():
            output = self.model(**tokenized, output_hidden_states=True)
        
        hidden_states = torch.cat(output[2])
        hidden_states = hidden_states.permute(1,0,2) # re-arrange dimensions: [token, layer, embedding_dim]
        
        return hidden_states
        