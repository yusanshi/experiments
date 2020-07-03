from config import model_name
import pandas as pd
import json
from tqdm import tqdm
from os import path
import random
from nltk.tokenize import word_tokenize
import numpy as np
import csv
from pathlib import Path
from shutil import copyfile
import importlib

try:
    Config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


def parse_behaviors(source, target, val_target, user2int_path):
    """
    Parse behaviors file in training set.
    Args:
        source: source behaviors file
        target: target behaviors file
        val_target: target behaviors file used for validation split from training set
        user2int_path: path for saving user2int file
    """
    print(f"Parse {source}")
    with open(source, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open(val_target, 'w') as f:
        f.writelines(lines[:int(len(lines) * Config.validation_proportion)])

    behaviors = pd.read_table(
        source,
        header=None,
        usecols=range(4),
        names=['user', 'time', 'clicked_news', 'impressions'])
    behaviors.clicked_news.fillna(' ', inplace=True)
    behaviors.impressions = behaviors.impressions.str.split()

    user2int = {}
    for row in behaviors.itertuples(index=False):
        if row.user not in user2int:
            user2int[row.user] = len(user2int) + 1

    pd.DataFrame(user2int.items(), columns=['user',
                                            'int']).to_csv(user2int_path,
                                                           sep='\t',
                                                           index=False)
    print(
        f'Please modify `num_users` in `src/config.py` into 1 + {len(user2int)}'
    )

    # Drop rows in val_behaviors
    val_behaviors = pd.read_table(val_target,
                                  header=None,
                                  usecols=range(2),
                                  names=['user', 'time'])
    behaviors['user-time'] = behaviors['user'] + behaviors['time']
    val_behaviors['user-time'] = val_behaviors['user'] + val_behaviors['time']
    behaviors.drop(behaviors[behaviors['user-time'].isin(
        val_behaviors['user-time'])].index,
                   inplace=True)
    print(
        f'Drop {len(val_behaviors)} sessions from training set to be used in validation.'
    )

    for row in behaviors.itertuples():
        behaviors.at[row.Index, 'user'] = user2int[row.user]

    with tqdm(total=len(behaviors), desc="Balancing data") as pbar:
        for row in behaviors.itertuples():
            positive = iter([x for x in row.impressions if x.endswith('1')])
            negative = [x for x in row.impressions if x.endswith('0')]
            random.shuffle(negative)
            negative = iter(negative)
            pairs = []
            try:
                while True:
                    pair = [next(positive)]
                    for _ in range(Config.negative_sampling_ratio):
                        pair.append(next(negative))
                    pairs.append(pair)
            except StopIteration:
                pass
            behaviors.at[row.Index, 'impressions'] = pairs
            pbar.update(1)
    behaviors = behaviors.explode('impressions').dropna(
        subset=["impressions"]).reset_index(drop=True)
    behaviors[['candidate_news', 'clicked']] = pd.DataFrame(
        behaviors.impressions.map(
            lambda x: (' '.join([e.split('-')[0] for e in x]), ' '.join(
                [e.split('-')[1] for e in x]))).tolist())
    behaviors.to_csv(
        target,
        sep='\t',
        index=False,
        columns=['user', 'clicked_news', 'candidate_news', 'clicked'])


def parse_news(source, target, category2int_path, word2int_path,
               entity2int_path, mode):
    """
    Parse news for training set and test set
    Args:
        source: source news file
        target: target news file
        if mode == 'train':
            category2int_path, word2int_path, entity2int_path: Path to save
        elif mode == 'test':
            category2int_path, word2int_path, entity2int_path: Path to load from
    """
    print(f"Parse {source}")
    news = pd.read_table(source,
                         header=None,
                         usecols=[0, 1, 2, 3, 4, 6, 7],
                         names=[
                             'id', 'category', 'subcategory', 'title',
                             'abstract', 'title_entities', 'abstract_entities'
                         ])
    news.title_entities.fillna('[]', inplace=True)
    news.abstract_entities.fillna('[]', inplace=True)
    news.fillna(' ', inplace=True)
    parsed_news = pd.DataFrame(columns=[
        'id', 'category', 'subcategory', 'title', 'abstract', 'title_entities',
        'abstract_entities'
    ])

    if mode == 'train':
        category2int = {}
        word2int = {}
        word2freq = {}
        entity2int = {}
        entity2freq = {}

        for row in news.itertuples(index=False):
            if row.category not in category2int:
                category2int[row.category] = len(category2int) + 1
            if row.subcategory not in category2int:
                category2int[row.subcategory] = len(category2int) + 1

            for w in word_tokenize(row.title.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1
            for w in word_tokenize(row.abstract.lower()):
                if w not in word2freq:
                    word2freq[w] = 1
                else:
                    word2freq[w] += 1

            for e in json.loads(row.title_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

            for e in json.loads(row.abstract_entities):
                times = len(e['OccurrenceOffsets']) * e['Confidence']
                if times > 0:
                    if e['WikidataId'] not in entity2freq:
                        entity2freq[e['WikidataId']] = times
                    else:
                        entity2freq[e['WikidataId']] += times

        for k, v in word2freq.items():
            if v >= Config.word_freq_threshold:
                word2int[k] = len(word2int) + 1

        for k, v in entity2freq.items():
            if v >= Config.entity_freq_threshold:
                entity2int[k] = len(entity2int) + 1

        with tqdm(total=len(news),
                  desc="Parsing categories, words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id,
                    category2int[row.category], category2int[row.subcategory],
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract,
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.title_entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]
                for e in json.loads(row.abstract_entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]

                try:
                    for i, w in enumerate(word_tokenize(row.title.lower())):
                        if w in word2int:
                            new_row[3][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[5][i] = local_entity_map[w]
                except IndexError:
                    pass

                try:
                    for i, w in enumerate(word_tokenize(row.abstract.lower())):
                        if w in word2int:
                            new_row[4][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[6][i] = local_entity_map[w]
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)

        parsed_news.to_csv(target, sep='\t', index=False)

        pd.DataFrame(category2int.items(),
                     columns=['category', 'int']).to_csv(category2int_path,
                                                         sep='\t',
                                                         index=False)
        print(
            f'Please modify `num_categories` in `src/config.py` into 1 + {len(category2int)}'
        )

        pd.DataFrame(word2int.items(), columns=['word',
                                                'int']).to_csv(word2int_path,
                                                               sep='\t',
                                                               index=False)
        print(
            f'Please modify `num_words` in `src/config.py` into 1 + {len(word2int)}'
        )

        pd.DataFrame(entity2int.items(),
                     columns=['entity', 'int']).to_csv(entity2int_path,
                                                       sep='\t',
                                                       index=False)
        print(
            f'Please modify `num_entities` in `src/config.py` into 1 + {len(entity2int)}'
        )

    elif mode == 'test':
        category2int = dict(pd.read_table(category2int_path).values.tolist())
        # na_filter=False is needed since nan is also a valid word
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())
        entity2int = dict(pd.read_table(entity2int_path).values.tolist())

        word_total = 0
        word_missed = 0

        with tqdm(total=len(news),
                  desc="Parsing categories, words and entities") as pbar:
            for row in news.itertuples(index=False):
                new_row = [
                    row.id, category2int[row.category] if row.category
                    in category2int else 0, category2int[row.subcategory]
                    if row.subcategory in category2int else 0,
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract,
                    [0] * Config.num_words_title,
                    [0] * Config.num_words_abstract
                ]

                # Calculate local entity map (map lower single word to entity)
                local_entity_map = {}
                for e in json.loads(row.title_entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]
                for e in json.loads(row.abstract_entities):
                    if e['Confidence'] > Config.entity_confidence_threshold and e[
                            'WikidataId'] in entity2int:
                        for x in ' '.join(e['SurfaceForms']).lower().split():
                            local_entity_map[x] = entity2int[e['WikidataId']]

                try:
                    for i, w in enumerate(word_tokenize(row.title.lower())):
                        word_total += 1
                        if w in word2int:
                            new_row[3][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[5][i] = local_entity_map[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass

                try:
                    for i, w in enumerate(word_tokenize(row.abstract.lower())):
                        word_total += 1
                        if w in word2int:
                            new_row[4][i] = word2int[w]
                            if w in local_entity_map:
                                new_row[6][i] = local_entity_map[w]
                        else:
                            word_missed += 1
                except IndexError:
                    pass

                parsed_news.loc[len(parsed_news)] = new_row

                pbar.update(1)

        print(f'Out-of-Vocabulary rate: {word_missed/word_total:.4f}')
        parsed_news.to_csv(target, sep='\t', index=False)

    else:
        print('Wrong mode!')


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
    val_dir = './data/val'
    test_dir = './data/test'
    Path(val_dir).mkdir(exist_ok=True)

    print('Process data for training')

    print('Parse behaviors')
    parse_behaviors(path.join(train_dir, 'behaviors.tsv'),
                    path.join(train_dir, 'behaviors_parsed.tsv'),
                    path.join(val_dir, 'behaviors.tsv'),
                    path.join(train_dir, 'user2int.tsv'))

    print('Parse news')
    parse_news(path.join(train_dir, 'news.tsv'),
               path.join(train_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='train')

    # For validation in training
    copyfile(path.join(train_dir, 'news_parsed.tsv'),
             path.join(val_dir, 'news_parsed.tsv'))

    print('Generate word embedding')
    generate_word_embedding(
        f'./data/glove/glove.840B.{Config.word_embedding_dim}d.txt',
        path.join(train_dir, 'pretrained_word_embedding.npy'),
        path.join(train_dir, 'word2int.tsv'))

    print('\nProcess data for evaluation')

    print('Parse news')
    parse_news(path.join(test_dir, 'news.tsv'),
               path.join(test_dir, 'news_parsed.tsv'),
               path.join(train_dir, 'category2int.tsv'),
               path.join(train_dir, 'word2int.tsv'),
               path.join(train_dir, 'entity2int.tsv'),
               mode='test')
