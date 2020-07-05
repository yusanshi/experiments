import copy
import numpy


def parse(results_path):
    lines = iter(open(results_path).read().splitlines())
    results = {}
    template = {
        'AUC': [],
        'MRR': [],
        'nDCG@5': [],
        'nDCG@10': [],
    }
    try:
        while True:
            line = next(lines)
            if line.startswith('round'):
                key = (int(line.split('-')[-2]), line.split('-')[-1] == '1')
                if key not in results:
                    results[key] = copy.deepcopy(template)
                while True:
                    line = next(lines)
                    if line.strip() == '':
                        break
                    else:
                        metrics = list(template.keys())
                        for metric in metrics:
                            if line.startswith(metric):
                                results[key][metric].append(
                                    float(line.split(':')[-1]))
                                break
    except StopIteration:
        pass

    for k in results.keys():
        for kk in results[k].keys():
            results[k][kk] = numpy.mean(results[k][kk]), numpy.var(
                results[k][kk])

    def flatten(two_dim):
        return [item for sublist in two_dim for item in sublist]

    metrics = list(template.keys())

    header = ['classification batch', 'joint loss'] + flatten(
        [[metric + '均值', metric + '方差'] for metric in metrics])
    print(' | '.join([''] + header + ['']))
    print(' | '.join([''] + ['----' for _ in range(len(header))] + ['']))
    for k, v in results.items():
        print(' | '.join([''] + [str(k[0]), str(k[1])] + flatten(
            [['{:.4f}'.format(v[metric][0]), '{:.6f}'.format(v[metric][1])]
             for metric in metrics]) + ['']))


if __name__ == "__main__":
    parse('results.txt')