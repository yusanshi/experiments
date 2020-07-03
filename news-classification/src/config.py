import os


class Config:
    num_batches = 800  # Number of batches to train
    num_batches_show_loss = 50  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 50
    batch_size = 32
    learning_rate = 0.001
    test_proportion = 0.2
    validation_proportion = 0.2
    num_workers = 4  # Number of workers for data loading
    # Whether try to load checkpoint
    load_checkpoint = os.environ['LOAD_CHECKPOINT'] == '1' if 'LOAD_CHECKPOINT' in os.environ else True
    num_urlparts_an_url = 8  # TODO
    num_words_a_sentence = 30  # TODO
    num_sentences_a_news = 10  # TODO
    num_words_a_news = 800  # used when not hierarchical
    word_freq_threshold = 3
    urlpart_freq_threshold = 2
    # Modify the following by the output of `src/dataprocess.py`
    num_urlparts = 1 + 1047
    num_words = 1 + 22195
    word_embedding_dim = 300
    window_size = 3
    query_vector_dim = 200
    num_attention_heads = 15

    use_url_text = os.environ['USE_URL_TEXT'] if 'USE_URL_TEXT' in os.environ else 'text'
    assert use_url_text in ['url', 'text', 'concatenate', 'attention']

    text_method = [os.environ['TEXT_METHOD_ONE'] if 'TEXT_METHOD_ONE' in os.environ else 'self-attention',
                   os.environ['TEXT_METHOD_TWO'] if 'TEXT_METHOD_TWO' in os.environ else 'average']
    assert text_method[0] in ['cnn', 'gru', 'lstm',
                              'self-attention', 'multihead-self-attention','none']
    assert text_method[1] in ['attention', 'average', 'maxpooling']
