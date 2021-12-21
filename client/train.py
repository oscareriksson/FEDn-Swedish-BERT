from __future__ import print_function
import sys
import yaml
import torch

from util.data_manager import ReviewsDataset, read_data
from util.transformations import np_to_weights, weights_to_np
from models.bert_pytorch_model import SentimentClassifier
from fedn.utils.pytorchhelper import PytorchHelper

from tqdm import tqdm


def train(model, data, loss_fn, optimizer, settings):
    print("-- RUNNING TRAINING --", flush=True)

    x_train, y_train = read_data(data, settings['nr_training_samples'])
    train_set = ReviewsDataset(x_train, y_train, settings['max_text_length'], settings['base_model'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings['batch_size'], num_workers=4)

    model.train()

    for i in range(settings['epochs']):
        for d in tqdm(train_loader):
            input_ids = d["input_ids"]
            attention_mask = d["attention_mask"]
            targets = d["targets"]

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            error = loss_fn(output, targets)
            error.backward()
            optimizer.step()

    print("-- TRAINING COMPLETED --", flush=True)
    return model


if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    helper = PytorchHelper()
    model = SentimentClassifier(settings['base_model'])
    model.load_state_dict(np_to_weights(helper.load_model(sys.argv[1])))

    model = train(model,
                  "../data/train.csv",
                  torch.nn.CrossEntropyLoss(),
                  torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7),
                  settings)

    helper.save_model(weights_to_np(model.state_dict()), sys.argv[2])
