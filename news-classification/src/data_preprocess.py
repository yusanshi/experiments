import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import re
import csv
from os import path
from pathlib import Path
from config import Config
import matplotlib.pyplot as plt


def show_hist(nums, title, percentile=90):
    plt.hist(nums, range(min(nums), int(0.95 * max(nums))))
    p = np.percentile(nums, percentile)
    plt.axvline(p, color='k', linestyle='dashed', linewidth=1)
    _, max_ylim = plt.ylim()
    plt.text(p * 1.2, max_ylim * 0.8, f'{percentile} Percentile: {p}')
    plt.title(title)
    plt.show()


def purify(source, target):
    df = pd.read_table(source)
    df['label'] = (df['label'] == '__label__evergreen').astype(int)

    def clean_url(url):
        # Remove url parameters since they are in a mess
        if url.find('?') != -1:
            url = url[:url.find('?')]

        url = url.strip()
        while url.startswith(r'\n') or url.startswith(r'\t'):
            url = url[2:]
        while url.endswith(r'\n') or url.endswith(r'\t'):
            url = url[:-2]
        url = url.strip()

        if url.startswith('http://'):
            url = url[len('http://'):]
        elif url.startswith('https://'):
            url = url[len('https://'):]
        else:
            print(f"Warning: {url}")

        return url

    df['url'] = df['url'].map(clean_url)

    def clean_text(text):
        # Remove url parameters in the begining of the text
        if text[0] == '?':
            text = text[text.find(' '):]
        text = text.strip()
        text = re.sub(r'[0-9]+ ', '', text)
        text = text.strip()
        return text

    df['text'] = df['text'].map(clean_text)

    df.to_csv(target, sep='\t', index=False)


def split(source, trian_target, test_target):
    df = pd.read_table(source)
    df = df.sample(frac=1).reset_index(drop=True)
    test_size = int(Config.test_proportion * len(df))
    test_df = df[:test_size].reset_index(drop=True)
    train_df = df[test_size:].reset_index(drop=True)

    test_df.to_csv(test_target, sep='\t', index=False)
    train_df.to_csv(trian_target, sep='\t', index=False)


def parse_news(source, target, urlpart2int_path, word2int_path, mode):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
    """
    print(f"Parse {source}")
    news = pd.read_table(source)
    parsed_news = pd.DataFrame(columns=[
        'url', 'text', 'label'
    ])

    if mode == 'train':
        urlpart2int = {}
        urlpart2freq = {}
        word2int = {}
        word2freq = {}

        num_words_a_sentence_list = []
        num_sentences_a_news_list = []
        num_words_a_news_list = []

        for row in news.itertuples(index=False):
            for p in row.url.lower().split('/'):
                if p not in urlpart2freq:
                    urlpart2freq[p] = 1
                else:
                    urlpart2freq[p] += 1
            num_sentences_a_news_list.append(
                len(sent_tokenize(row.text.lower())))
            temp_list = []
            for sent in sent_tokenize(row.text.lower()):
                num_words_a_sentence_list.append(len(word_tokenize(sent)))
                temp_list.append(len(word_tokenize(sent)))
                for w in word_tokenize(sent):
                    if w not in word2freq:
                        word2freq[w] = 1
                    else:
                        word2freq[w] += 1
            num_words_a_news_list.append(sum(temp_list))

        show_hist(num_words_a_sentence_list, 'num_words_a_sentence_list')
        show_hist(num_sentences_a_news_list, 'num_sentences_a_news_list')
        show_hist(num_words_a_news_list, 'num_words_a_news_list')

        for k, v in urlpart2freq.items():
            if v >= Config.urlpart_freq_threshold:
                urlpart2int[k] = len(urlpart2int) + 1
        for k, v in word2freq.items():
            if v >= Config.word_freq_threshold:
                word2int[k] = len(word2int) + 1
    else:
        urlpart2int = dict(pd.read_table(
            urlpart2int_path, na_filter=False).values.tolist())
        word2int = dict(pd.read_table(
            word2int_path, na_filter=False).values.tolist())

    urlpart_total = 0
    urlpart_missed = 0
    word_total = 0
    word_missed = 0

    with tqdm(total=len(news),
              desc="Parsing urlparts and words") as pbar:
        for row in news.itertuples(index=False):

            new_row = [
                [],
                [],
                row.label
            ]

            for p in row.url.lower().split('/'):
                urlpart_total += 1
                if p in urlpart2int:
                    new_row[0].append(urlpart2int[p])
                else:
                    new_row[0].append(0)
                    urlpart_missed += 1

            for sent in sent_tokenize(row.text.lower()):
                new_row[1].append([])
                for w in word_tokenize(sent):
                    word_total += 1
                    if w in word2int:
                        new_row[1][-1].append(word2int[w])
                    else:
                        new_row[1][-1].append(0)
                        word_missed += 1

            parsed_news.loc[len(parsed_news)] = new_row

            pbar.update(1)
    print(
        f'Urlpart Out-of-Vocabulary rate in {mode} set: {urlpart_missed/ urlpart_total:.4f}')
    print(
        f'Word Out-of-Vocabulary rate in {mode} set: {word_missed/ word_total:.4f}')
    parsed_news.to_csv(target, sep='\t', index=False)

    if mode == 'train':
        pd.DataFrame(urlpart2int.items(), columns=['urlpart',
                                                   'int']).to_csv(urlpart2int_path,
                                                                  sep='\t',
                                                                  index=False)
        print(
            f'Please modify `num_urlparts` in `src/config.py` into 1 + {len(urlpart2int)}'
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'
        )


def generate_word_embedding(source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, initial its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    # na_filter=False is needed since nan is also a valid word
    word2int = dict(
        pd.read_table(word2int_path, na_filter=False).values.tolist())
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE)
    target_embedding = np.random.normal(size=(1 + len(word2int),
                                              Config.word_embedding_dim))
    target_embedding[0] = 0
    word_missed = 0
    with tqdm(total=len(word2int),
              desc="Generating word embedding from pretrained embedding file"
              ) as pbar:
        for k, v in word2int.items():
            if k in source_embedding.index:
                target_embedding[v] = source_embedding.loc[k].tolist()
            else:
                word_missed += 1

            pbar.update(1)

    print(
        f'Rate of word missed in pretrained embedding: {word_missed/len(word2int):.4f}'
    )
    np.save(target, target_embedding)


if __name__ == '__main__':
    train_dir = './data/train'
    test_dir = './data/test'
    Path(train_dir).mkdir(exist_ok=True)
    Path(test_dir).mkdir(exist_ok=True)

    purify('data/data_all_withURL.2020-05-26.tsv', 'data/news_purified.tsv')

    split('data/news_purified.tsv',
          path.join(train_dir, 'news_purified.tsv'),
          path.join(test_dir, 'news_purified.tsv'),
          )

    parse_news(path.join(train_dir, 'news_purified.tsv'),
               path.join(train_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'urlpart2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               'train'
               )

    generate_word_embedding(
        f'./data/glove/glove.840B.{Config.word_embedding_dim}d.txt',
        path.join(train_dir, 'pretrained_word_embedding.npy'),
        path.join(train_dir, 'word2int.tsv'))

    parse_news(path.join(test_dir, 'news_purified.tsv'),
               path.join(test_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'urlpart2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               'test'
               )
