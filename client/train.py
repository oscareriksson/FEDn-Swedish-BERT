from __future__ import print_function
import sys
import yaml
import torch

from util.data_manager import ReviewsDataset, read_data
from util.transformations import np_to_weights, weights_to_np
from models.bert_pytorch_model import SentimentClassifier
from fedn.utils.pytorchhelper import PytorchHelper

from tqdm import tqdm


def train(model, device, train_loader, optimizer, settings):
    print("-- RUNNING TRAINING --", flush=True)

    model.train()

    for i in range(settings['epochs']):
        for d in tqdm(train_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            optimizer.zero_grad()
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            error = torch.nn.CrossEntropyLoss()(output, targets)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentClassifier(settings['base_model']).to(device)
    model.load_state_dict(np_to_weights(helper.load_model(sys.argv[1])))

    x_train, y_train = read_data("../data/train.csv", settings['nr_training_samples'])
    train_set = ReviewsDataset(x_train, y_train, settings['max_text_length'], settings['base_model'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings['batch_size'], num_workers=4)

    model = train(model,
                  device,
                  train_loader,
                  torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7),
                  settings)

    helper.save_model(weights_to_np(model.state_dict()), sys.argv[2])
