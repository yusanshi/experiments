import json
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def analyse(directory, granularity=0.025):
    prediction_path = path.join(directory, 'prediction-TANR.txt')
    behaviors_path = path.join(directory, 'behaviors.tsv')
    news_parsed_path = path.join(directory, 'news_parsed.tsv')

    behaviors = pd.read_table(
        behaviors_path,
        header=None,
        usecols=[0, 3, 4],
        index_col='impression_id',
        names=['impression_id', 'clicked_news', 'candidate_news'])
    behaviors.clicked_news.fillna(' ', inplace=True)

    news_parsed = pd.read_table(news_parsed_path, usecols=range(3))
    newsid2category = dict(news_parsed[['id', 'category']].values)
    newsid2subcategory = dict(news_parsed[['id', 'subcategory']].values)

    assert granularity > 0 and granularity < 1
    assert int(1 / granularity) == 1 / granularity
    rank2category_same_rate = {i: [] for i in range(int(1 / granularity))}
    rank2subcategory_same_rate = {i: [] for i in range(int(1 / granularity))}

    with open(prediction_path) as f:
        lines = f.read().splitlines()
        with tqdm(total=len(lines)) as pbar:
            for line in lines:
                assert len(line.split()) == 2
                impression_id, ranks = line.split()
                impression_id = int(impression_id)
                clicked_news = behaviors.loc[impression_id].clicked_news.split(
                )
                if len(clicked_news) == 0:
                    continue
                categories = [newsid2category[x] for x in clicked_news]
                subcategories = [newsid2subcategory[x] for x in clicked_news]
                ranks = json.loads(ranks)
                if max(ranks) != len(ranks):
                    continue
                ranks = [x / len(ranks) for x in ranks]
                for i, rank in enumerate(ranks):
                    key = int(rank / granularity)
                    if key in rank2category_same_rate and key in rank2subcategory_same_rate:
                        candidate_news = behaviors.loc[
                            impression_id].candidate_news.split()[i].split(
                                '-')[0]
                        rank2category_same_rate[key].append(
                            categories.count(newsid2category[candidate_news]) /
                            len(categories))
                        rank2subcategory_same_rate[key].append(
                            subcategories.count(
                                newsid2subcategory[candidate_news]) /
                            len(subcategories))

                pbar.update(1)

    rank2category_same_rate = {
        k * granularity: np.mean(v)
        for k, v in rank2category_same_rate.items()
    }
    rank2subcategory_same_rate = {
        k * granularity: np.mean(v)
        for k, v in rank2subcategory_same_rate.items()
    }

    plt.title('Rank - (sub)category coincidence rate')
    plt.xlabel('Rank (scaled to 1)')
    plt.plot(list(rank2category_same_rate.keys()),
             list(rank2category_same_rate.values()),
             label="Category coincidence rate")
    plt.plot(list(rank2subcategory_same_rate.keys()),
             list(rank2subcategory_same_rate.values()),
             label="Subategory coincidence rate")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    analyse('data/test')