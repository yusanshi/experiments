import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from config import Config
from torch.utils.data import DataLoader
from dataset import MyDataset
from model import Model
from sklearn.metrics import classification_report

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, dataset):
    """
    Evaluate model on target dataset.
    Args:
        model: model to be evaluated
        dataset:
    Returns:
        loss
        A dict, see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    dataloader = iter(
        DataLoader(dataset,
                   batch_size=Config.batch_size,
                   shuffle=False,
                   num_workers=Config.num_workers,
                   drop_last=False))

    criterion = nn.BCEWithLogitsLoss()
    y_pred_full = []
    y_full = []
    loss_full = []
    with tqdm(total=len(dataloader), desc="Evaluating") as pbar:
        for minibatch in dataloader:
            # batch_size
            y_pred = model(minibatch)
            # batch_size
            y = minibatch['label'].float()
            loss = criterion(y_pred, y.to(device))
            loss_full.append(loss.item())
            y_pred_full.extend((torch.sigmoid(y_pred) > 0.5).int().tolist())
            y_full.extend(y.tolist())

            pbar.update(1)

    return np.mean(loss_full), classification_report(y_full, y_pred_full,
                                                     output_dict=True, zero_division=0)


if __name__ == '__main__':
    test_dataset = MyDataset('data/test/news_parsed.tsv')
    # Don't need to load pretrained word embedding
    # since it will be loaded from checkpoint later
    model = Model(Config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint('./checkpoint')
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    loss, report = evaluate(model, test_dataset)
    print(report['weighted avg'])
