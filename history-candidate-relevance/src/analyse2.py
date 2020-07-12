import json
from os import path
import pandas as pd


def analyse(directory):
    prediction_path = path.join(directory, 'prediction-TANR.txt')
    behaviors_path = path.join(directory, 'behaviors.tsv')
    news_path = path.join(directory, 'news.tsv')
    behaviors = pd.read_table(
        behaviors_path,
        header=None,
        usecols=[0, 3, 4],
        index_col='impression_id',
        names=['impression_id', 'clicked_news', 'candidate_news'])
    behaviors.clicked_news.fillna(' ', inplace=True)

    news = pd.read_table(
        news_path,
        header=None,
        usecols=range(5),
        index_col='id',
        names=['id', 'category', 'subcategory', 'title', 'abstract'])

    count = 0

    with open(prediction_path) as f:
        for i, line in enumerate(f.read().splitlines()):
            assert len(line.split()) == 2
            impression_id, ranks = line.split()
            ranks = json.loads(ranks)
            if max(ranks) != len(ranks) or len(ranks) < 293:
                continue
            impression_id = int(impression_id)
            clicked_news = behaviors.loc[impression_id].clicked_news.split()
            if len(clicked_news) == 0:
                continue
            candidate_news = behaviors.loc[impression_id].candidate_news.split(
            )[ranks.index(1)].split('-')[0]
            count += 1
            print(f"## Pair {count}")
            print("| id | category  | subcategory | title | abstract |")
            print("| --- | ---  | --- | --- | --- |")
            for x in [candidate_news] + clicked_news:
                row = news.loc[x]
                print(
                    f"| {x} | {row.category} | {row.subcategory} | {row.title} | {row.abstract} |"
                )
            print()
            print()
            print()


if __name__ == "__main__":
    analyse('data/test')