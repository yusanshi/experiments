from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from config import Config


class MyDataset(Dataset):
    def __init__(self, news_path):
        super(MyDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            converters={
                'url': literal_eval,
                'text': literal_eval
            })

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        row = self.news_parsed.iloc[idx]

        def flatten(two_dim_list):
            return [j for sub in two_dim_list for j in sub]

        def pad_one_layer(seq, max_length):
            seq = seq[:max_length]
            if len(seq) < max_length:
                seq += [0] * (max_length - len(seq))
            return seq

        def pad_two_layer(seq, fist_max_length, second_max_length):
            seq = seq[:fist_max_length]
            if len(seq) < fist_max_length:
                seq += [[]] * (fist_max_length - len(seq))
            seq = [pad_one_layer(x, second_max_length) for x in seq]
            return seq

        item = {
            "url": pad_one_layer(row.url, Config.num_urlparts_an_url),
            "news": {
                "hierarchical": pad_two_layer(row.text, Config.num_sentences_a_news, Config.num_words_a_sentence),
                "flatten": pad_one_layer(flatten(row.text), Config.num_words_a_news),
            },
            "label": row.label
        }

        return item
