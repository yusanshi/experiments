import torch
import torch.nn.functional as F


class SelfAttention(torch.nn.Module):
    """
    A general self attention module.
    Originally for Hi-Fi Ark.
    """
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, candidate_vector, length=None):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
            length: batch_size
        Returns:
            (shape) batch_size, candidate_size, candidate_vector_dim
        """
        # batch_size, candidate_size, candidate_size
        weights = torch.bmm(candidate_vector, candidate_vector.transpose(1, 2))

        if length is not None:
            batch_size, maxlen, _ = weights.size()
            mask = torch.arange(maxlen).expand(batch_size, maxlen,
                                               maxlen) < length.view(-1, 1, 1)
            weights[~mask] = float('-inf')

        weights = F.softmax(weights, dim=2)

        # batch_size, candidate_size, candidate_vector_dim
        self_attended_vector = torch.bmm(weights, candidate_vector)
        return self_attended_vector
