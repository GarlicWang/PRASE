from transformers import BertJapaneseTokenizer, BertModel
import torch

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            print("device : ", device)
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
    def __init__(self):
        self.model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens")

    # prefix format : f123_filmarksTitle, s456_sakuTitle
    def get_sbert_dict(self, kg1_head_set, kg2_head_set, remove_prefix=False):
        kg1_sbert_dict, kg2_sbert_dict = dict(), dict()
        kg1_title_list, kg2_title_list = list(kg1_head_set), list(kg2_head_set)
        if remove_prefix:
            kg1_noprefix_title_list, kg2_noprefix_title_list = [], []
            for title in kg1_title_list:
                title = title.lstrip('f')
                for i, c in enumerate(title):
                    if c not in '0123456789':
                        title = title[i:]
                        break
                parsed_title = title.lstrip('_')
                kg1_noprefix_title_list.append(parsed_title)
            for title in kg2_title_list:
                title = title.lstrip('s')
                for i, c in enumerate(title):
                    if c not in '0123456789':
                        title = title[i:]
                        break
                parsed_title = title.lstrip('_')
                kg2_noprefix_title_list.append(parsed_title)
            self.kg1_emb = self.model.encode(kg1_noprefix_title_list)
            self.kg2_emb = self.model.encode(kg2_noprefix_title_list)
        else:
            self.kg1_emb = self.model.encode(kg1_title_list)
            self.kg2_emb = self.model.encode(kg2_title_list)
        for i, title in enumerate(kg1_title_list):
            kg1_sbert_dict[title] = self.kg1_emb[i].tolist()
        for i, title in enumerate(kg2_title_list):
            kg2_sbert_dict[title] = self.kg2_emb[i].tolist()
        return kg1_sbert_dict, kg2_sbert_dict
