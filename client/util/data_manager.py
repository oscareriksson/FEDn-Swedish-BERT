import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd


class ReviewsDataset(Dataset):
    def __init__(self, reviews, targets, max_length, base_model):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer(review, max_length=self.max_length, padding='max_length', truncation=True,
                                  return_tensors="pt")

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def read_data(filename, nr_samples):
    """
    Helper function to read util and split it.
    :return: train and test util
    """

    print("-- START READING DATA --")
    df = pd.read_csv(filename)[:nr_samples]

    x = df.text.to_numpy()
    y = df.label.to_numpy()

    print("Using {} samples".format(y.shape[0]))
    return x, y
