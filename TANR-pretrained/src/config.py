import os

model_name = 'TANR'


class BaseConfig():
    """
    General configurations appiled to all models
    """
    num_batches = 8000  # Number of batches to train
    num_batches_show_loss = 50  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 200
    batch_size = 128
    learning_rate = 0.001
    validation_proportion = 0.1
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 3
    entity_freq_threshold = 3
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 4  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 31312
    num_categories = 1 + 274
    num_entities = 1 + 8312
    num_users = 1 + 50000
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1

    num_batches_classification = 1000
    classification_initiate = os.environ[
        'CLASSIFICATION_INITIATE'] == '1' if 'CLASSIFICATION_INITIATE' in os.environ else True
    joint_loss = os.environ[
        'JOINT_LOSS'] == '1' if 'JOINT_LOSS' in os.environ else True
