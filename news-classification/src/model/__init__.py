import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention.additive import AdditiveAttention
from model.attention.self import SelfAttention
from model.attention.multihead_self import MultiHeadSelfAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(Model, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        self.urlpart_embedding = nn.Embedding(config.num_urlparts,
                                              config.word_embedding_dim, padding_idx=0)
        self.url_additive_attention = AdditiveAttention(
            config.query_vector_dim, config.word_embedding_dim)

        self.news_cnn = nn.Conv2d(
            1,
            config.word_embedding_dim,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.news_self_attention = SelfAttention()
        self.news_additive_attention = AdditiveAttention(
            config.query_vector_dim, config.word_embedding_dim)
        self.news_multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads)
        self.gru = nn.GRU(config.word_embedding_dim,
                          config.word_embedding_dim, batch_first=True)
        self.lstm = nn.LSTM(config.word_embedding_dim,
                            config.word_embedding_dim, batch_first=True)

        self.final_additive_attention = AdditiveAttention(
            config.query_vector_dim, config.word_embedding_dim)
        self.final_linear = nn.Linear(
            2 * config.word_embedding_dim if config.use_url_text == 'concatenate' else config.word_embedding_dim, 1)

    def forward(self, minibatch):
        """
        Args:
            minibatch:
                {
                    "url": Tensor(batch_size) * num_urlparts_an_url,
                    "news": {
                        "hierarchical": [Tensor(batch_size) * num_words_a_sentence] * num_sentences_a_news ,
                        "flatten": Tensor(batch_size) * num_words_a_news,
                    }
                }
        Returns:
            classification: batch_size
        """
        if self.config.use_url_text != 'text':
            # batch_size, num_urlparts_an_url, word_embedding_dim
            url_vector = self.urlpart_embedding(
                torch.stack(minibatch["url"], dim=1).to(device))
            # batch_size, word_embedding_dim
            url_vector_attended = self.url_additive_attention(url_vector)

        if self.config.use_url_text != 'url':
            # batch_size, num_words_a_news, word_embedding_dim
            news_vector = self.word_embedding(
                torch.stack(minibatch["news"]["flatten"], dim=1).to(device))

            if self.config.text_method[0] == 'cnn':
                # batch_size, word_embedding_dim, num_words_a_news
                convoluted_news_vector = self.news_cnn(
                    news_vector.unsqueeze(dim=1)).squeeze(dim=3)
                # batch_size, num_words_a_news, word_embedding_dim,
                first_news_vector = F.relu(
                    convoluted_news_vector).transpose(1, 2)
            elif self.config.text_method[0] == 'gru':
                # batch_size， num_words_a_news, word_embedding_dim
                temp, _ = self.gru(news_vector)
                # batch_size, num_words_a_news, word_embedding_dim,
                first_news_vector = temp
            elif self.config.text_method[0] == 'lstm':
                # batch_size， num_words_a_news, word_embedding_dim
                temp, _ = self.lstm(news_vector)
                # batch_size, num_words_a_news, word_embedding_dim,
                first_news_vector = temp
            elif self.config.text_method[0] == 'self-attention':
                # batch_size, num_words_a_news, word_embedding_dim
                first_news_vector = self.news_self_attention(news_vector)
            elif self.config.text_method[0] == 'multihead-self-attention':
                # batch_size, num_words_a_news, word_embedding_dim
                first_news_vector = self.news_multihead_self_attention(
                    news_vector)
            elif self.config.text_method[0] == 'none':
                first_news_vector = news_vector

            if self.config.text_method[1] == 'attention':
                # batch_size, word_embedding_dim
                final_news_vector = self.news_additive_attention(
                    first_news_vector)
            elif self.config.text_method[1] == 'average':
                # batch_size, word_embedding_dim
                final_news_vector = first_news_vector.mean(dim=1)
            elif self.config.text_method[1] == 'maxpooling':
                # batch_size, word_embedding_dim
                final_news_vector = F.max_pool1d(
                    first_news_vector.transpose(1, 2), kernel_size=first_news_vector.size(1)).squeeze(dim=2)

        if self.config.use_url_text == 'url':
            return self.final_linear(url_vector_attended).squeeze(dim=1)
        elif self.config.use_url_text == 'text':
            return self.final_linear(final_news_vector).squeeze(dim=1)
        elif self.config.use_url_text == 'concatenate':
            return self.final_linear(torch.cat((url_vector_attended, final_news_vector), dim=1)).squeeze(dim=1)
        elif self.config.use_url_text == 'attention':
            # batch_size, word_embedding_dim
            temp = self.final_additive_attention(torch.stack(
                (url_vector_attended, final_news_vector), dim=1))
            return self.final_linear(temp).squeeze(dim=1)
