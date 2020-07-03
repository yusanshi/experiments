from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn.functional as F
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import importlib
import datetime

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    Config = getattr(importlib.import_module('config'), f"{model_name}Config")
except (AttributeError, ModuleNotFoundError):
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def train():
    writer = SummaryWriter(
        log_dir=
        f"./runs/{model_name}/{datetime.datetime.now().replace(microsecond=0).isoformat()}-{Config.classification_initiate}-{Config.joint_loss}"
    )

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load('./data/train/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None

    model = Model(Config, pretrained_word_embedding, writer).to(device)

    print(model)

    dataset = BaseDataset('data/train/behaviors_parsed.tsv',
                          'data/train/news_parsed.tsv',
                          Config.dataset_attributes)

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=Config.batch_size,
                   shuffle=True,
                   num_workers=Config.num_workers,
                   drop_last=True))

    checkpoint_dir = os.path.join('./checkpoint', model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if Config.classification_initiate:
        step_classification = 0
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=Config.learning_rate)
        start_time = time.time()
        with tqdm(total=Config.num_batches_classification,
                  desc="Training (classification)") as pbar:
            for i in range(1, Config.num_batches_classification + 1):
                try:
                    minibatch = next(dataloader)
                except StopIteration:
                    dataloader = iter(
                        DataLoader(dataset,
                                   batch_size=Config.batch_size,
                                   shuffle=True,
                                   num_workers=Config.num_workers,
                                   drop_last=True))
                    minibatch = next(dataloader)

                step_classification += 1

                _, topic_classification_loss = model(
                    minibatch["candidate_news"],
                    minibatch["clicked_news"],
                    classification_only=True)

                writer.add_scalar('Train(classification)/Loss',
                                  topic_classification_loss.item(),
                                  step_classification)
                optimizer.zero_grad()
                topic_classification_loss.backward()
                optimizer.step()

                if i % Config.num_batches_show_loss == 0:
                    tqdm.write(
                        f"Time {time_since(start_time)}, batches {i}, current loss {topic_classification_loss.item():.4f}"
                    )

                pbar.update(1)

    loss_full = []
    exhaustion_count = 0
    step = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    start_time = time.time()

    early_stopping = EarlyStopping()

    with tqdm(total=Config.num_batches, desc="Training") as pbar:
        for i in range(1, Config.num_batches + 1):
            try:
                minibatch = next(dataloader)
            except StopIteration:
                exhaustion_count += 1
                tqdm.write(
                    f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
                )
                dataloader = iter(
                    DataLoader(dataset,
                               batch_size=Config.batch_size,
                               shuffle=True,
                               num_workers=Config.num_workers,
                               drop_last=True))
                minibatch = next(dataloader)

            step += 1

            y_pred, topic_classification_loss = model(
                minibatch["candidate_news"], minibatch["clicked_news"])
            loss = torch.stack([x[0] for x in -F.log_softmax(y_pred, dim=1)
                                ]).mean()
            if Config.joint_loss:
                writer.add_scalar('Train/BaseLoss', loss.item(), step)
                writer.add_scalar('Train/TopicClassificationLoss',
                                  topic_classification_loss.item(), step)
                writer.add_scalar(
                    'Train/TopicBaseRatio',
                    topic_classification_loss.item() / loss.item(), step)
                loss += Config.topic_classification_loss_weight * topic_classification_loss
            loss_full.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train/Loss', loss.item(), step)

            if i % Config.num_batches_show_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}"
                )

            if i % Config.num_batches_validate == 0:
                val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                    model, './data/val')
                writer.add_scalar('Validation/AUC', val_auc, step)
                writer.add_scalar('Validation/MRR', val_mrr, step)
                writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
                writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
                )

                early_stop, get_better = early_stopping(-val_auc)
                if early_stop:
                    tqdm.write('Early stop.')
                    break
                elif get_better:
                    torch.save({'model_state_dict': model.state_dict()},
                               f"./checkpoint/{model_name}/ckpt-{step}.pth")

            pbar.update(1)


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model {model_name}')
    train()
