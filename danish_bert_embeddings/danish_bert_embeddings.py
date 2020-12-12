import os
import torch
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import AutoModelForPreTraining


class DanishBertEmbeddings():
    """Class for creating Danish BERT word and sentence on the fly. """

    
    def __init__(self):
        """
        Parameters
        ----------
        bert_path : str 
            Path to danish bert model. Default: ./bert-base-danish        
        """
        super().__init__()
    
        self.tokenizer = BertTokenizer.from_pretrained("DJSammy/bert-base-danish-uncased_BotXO,ai")
        self.model = AutoModelForPreTraining.from_pretrained("DJSammy/bert-base-danish-uncased_BotXO,ai")
    

    def embed(self, text, output_numpy=True):
        
        tokenized = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            output = self.model(**tokenized, output_hidden_states=True)
        
        hidden_states = torch.cat(output[2])

        # embedding method (one should be able to switch method to test different approaches)
        x = hidden_states[-2:,:,:] # pick last X layers
        x = x.sum(1) # sum token dimension 
        embedding = x.mean(0) # average layer dimension
        
        if output_numpy:
            embedding = embedding.numpy()
        
        embedding = embedding.reshape(1,-1)
        
        return embedding
        
        

    # def check_model_files(self):
    #     print('Checking required model files..')
    #     if os.path.isdir(self.bert_path):
    #         pt_check = all([file in pt_required_files for file in os.listdir(self.bert_path)])
    #         tf_check = all([file in tf_required_files for file in os.listdir(self.bert_path)])
    #     else:
    #         print('Model was not found..')
    #         pt_check = False
    #         tf_check = False

    #     return pt_check, tf_check
    
            
    # def load_bert(self):
    #     print('Loading model..')
    #     self.config = BertConfig.from_json_file(os.path.join(self.bert_path, 'bert-base-danish/config.json'))
    #     self.tokenizer = BertTokenizer(vocab_file=os.path.join(self.bert_path, 'bert-base-danish/vocab.txt'))
    #     self.model = BertModel(config=self.config).from_pretrained(
    #         os.path.join(self.bert_path, 'bert-base-danish/config.json'),
    #     )
    #     self.model.eval()
    #     print('Ready for creating awesome embeddings!')
    
    # def convert_from_tensorflow_to_pytorch(self):
    #     if 'pytorch_model_model.bin' in os.listdir(os.path.join(self.bert_path, 'bert-base-danish')):
    #         print('Model is already converted.')
    #     else:
    #         print('Converting model to PyTorch..')
    #         os.system('transformers-cli convert --model_type bert --tf_checkpoint {} --config {} --pytorch_dump_output {}'.format(os.path.join(self.bert_path, 'bert-base-danish/bert_model.ckpt'), os.path.join(self.bert_path, 'bert-base-danish/config.json'), os.path.join(self.bert_path, 'bert-base-danish/pytorch_model.bin')))
    #         # os.system('transformers-cli convert --model_type bert --tf_checkpoint {} --config {} --pytorch_dump_output {}'.format(os.path.join(self.bert_path, 'bert-base-danish/bert_model.ckpt'), os.path.join(self.bert_path, 'bert-base-danish/bert_config.json'), os.path.join(self.bert_path, 'bert-base-danish/pytorch_model.bin')))
    #         # os.system('mv {} {}'.format(os.path.join(self.bert_path, './bert-base-danish/bert_config.json'), os.path.join(self.bert_path, './bert-base-danish/config.json')))
    #     return None
    
    # def download_danish_bert(self):

    #     if os.path.isdir(os.path.join(self.bert_path, 'bert-base-danish')):
    #         print('Model dir already exists.')
    #     else: 
    #         print('Downloading danish BERT (this may take some time ^^)..')
    #         r = requests.get(self.url, allow_redirects=True)
    #         open('bert-base-danish.zip', 'wb').write(r.content)
    #         os.system('unzip bert-base-danish.zip')
    #         os.system('mv danish_bert_uncased_v2 bert-base-danish')
    #         os.system('rm bert-base-danish.zip')
    #         os.system('rm -rf __MACOSX')
    #         self.convert_from_tensorflow_to_pytorch()
    #         return None


