import torch
from model.general.attention.multihead_self import MultiHeadSelfAttention
from model.general.attention.additive import AdditiveAttention


class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)

    def forward(self, clicked_news_vector, clicked_news_length):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
            clicked_news_length: batch_size
        Returns:
            (shape) batch_size,  word_embedding_dim
        """
        # batch_size, word_embedding_dim
        final_user_vector = self.additive_attention(
            clicked_news_vector,
            clicked_news_length if self.config.use_mask else None)
        return final_user_vector
