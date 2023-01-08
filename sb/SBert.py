from transformers import BertJapaneseTokenizer, BertTokenizer, BertModel
import torch
import requests
from tqdm import tqdm
import pickle
    
class SentenceBert:
    def __init__(self, model_name_or_path, lang):
        if lang == "jp":
            self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        elif lang == "en":
            self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
        else:
            raise BaseException("No such language")
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        device = "cuda:2" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                        truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)

class SBert:
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset in ["dbp_wd_15k_V1", "dbp_yg_15k_V1"]:
            self.model = SentenceBert("bert-base-cased", "en")
        elif dataset in ["KKS"]:
            self.model = SentenceBert("sonoisa/sentence-bert-base-ja-mean-tokens", "en")

    def get_sbert_dict(self, kg1_head_set, kg2_head_set):
        kg1_sbert_dict, kg2_sbert_dict = dict(), dict()
        kg1_head_list, kg2_head_list = list(kg1_head_set), list(kg2_head_set)
        if self.dataset == "dbp_wd_15k_V1":
            kg1_head_text_list = [head.split('/')[-1] for head in kg1_head_list]  # kg1 : DBpedia
            kg2_head_id_list = [head.split('/')[-1] for head in kg2_head_list]  # kg2 : Wikidata
            with open("/tmp2/yhwang/EA_dataset/DWY15K/dbp_wd_15k_V1/wiki_id_label_dict.pkl", "rb") as f:
                id_label_dict = pickle.load(f)
            kg2_head_text_list = [id_label_dict[id] for id in kg2_head_id_list]
        
        elif self.dataset == "dbp_yg_15k_V1":
            kg1_head_text_list = [head.split('/')[-1] for head in kg1_head_list]  # kg1 : DBpedia
            kg2_head_text_list = kg2_head_list  # kg2 : YAGO
        
        elif self.dataset == "KKS":
            kg1_head_text_list = kg1_head_list
            kg2_head_text_list = kg2_head_list
        
        else:
            raise BaseException("Invalid Dataset!")
        
        self.kg1_emb = self.model.encode(kg1_head_text_list)
        self.kg2_emb = self.model.encode(kg2_head_text_list)
        
        for i, head in enumerate(kg1_head_list):
            kg1_sbert_dict[head] = self.kg1_emb[i].tolist()
        for i, head in enumerate(kg2_head_list):
            kg2_sbert_dict[head] = self.kg2_emb[i].tolist()
        return kg1_sbert_dict, kg2_sbert_dict
