from transformers import BertJapaneseTokenizer, BertModel
import torch

class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda:1" if torch.cuda.is_available() else "cpu"
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
        self.vod = list()
        self.tv = list()
        # self.vod_emb = list()
        # self.tv_emb = list()

    def load_triple(self, rel_path_1, rel_path_2):
        vod, tv = set(), set()
        with open(rel_path_1, "r") as f:
            data = f.readlines()
            for row in data:
                vod.add(row.strip().split('\t')[0])

        with open(rel_path_2, "r") as f:
            data = f.readlines()
            for row in data:
                tv.add(row.strip().split('\t')[0])
        self.vod = list(vod)
        self.tv = list(tv)

    def get_sbert_dict(self, rel_path_1, rel_path_2):
        kg1_sbert_dict, kg2_sbert_dict = dict(), dict()
        self.load_triple(rel_path_1, rel_path_2)
        # print("Japanese Sentence BERT is inferencing...")
        self.vod_emb = self.model.encode(self.vod)
        self.tv_emb = self.model.encode(self.tv)
        for i in range(len(self.vod)):
            kg1_sbert_dict[self.vod[i]] = self.vod_emb[i].tolist()
        for i in range(len(self.tv)):
            kg2_sbert_dict[self.tv[i]] = self.tv_emb[i].tolist()
        return kg1_sbert_dict, kg2_sbert_dict
