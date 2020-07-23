import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def analyse(prediction_path, length=2):
    ranks_absolute = []
    ranks_relative = []
    with open(prediction_path) as f:
        for line in f:
            assert len(line.strip('\n').split()) == 3
            hist_len = int(line.strip('\n').split()[1])
            if hist_len != length:
                continue
            ranks = json.loads(line.strip('\n').split()[2])
            ranks_absolute.append(ranks[0])
            ranks_relative.append(ranks[0] / len(ranks))

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(hspace=0.7)

    density = stats.gaussian_kde(ranks_relative)
    _, x, _ = ax1.hist(ranks_relative,
                       100,
                       rwidth=0.8,
                       weights=np.ones(len(ranks_relative)) /
                       len(ranks_relative))
    ax1.plot(x, density(x) / 100)
    ax1.set_xlabel('Rank (scaled to 1)')
    ax1.set_ylabel('Percentage')
    ax1.set_title('Rank distribution of padded news (relative)')

    density = stats.gaussian_kde(ranks_absolute)
    _, x, _ = ax2.hist(ranks_absolute,
                       range(1, 30),
                       rwidth=0.7,
                       weights=np.ones(len(ranks_absolute)) /
                       len(ranks_absolute))
    ax2.plot(x, density(x))
    ax2.set_xlabel('Rank (unscaled)')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Rank distribution of padded news (absolute)')

    plt.savefig(f'figs/history-{length}.png')


if __name__ == "__main__":
    for i in range(1, 51):
        analyse('data/test/prediction.txt', i)