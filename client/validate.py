import sys
import json
import yaml
import torch

from util.data_manager import ReviewsDataset, read_data
from util.transformations import np_to_weights
from fedn.utils.pytorchhelper import PytorchHelper
from models.bert_pytorch_model import SentimentClassifier

from tqdm import tqdm


def validate(model, loss_function, test_data, train_data, settings):
    print("-- RUNNING VALIDATION --", flush=True)

    def evaluate(data_loader):
        model.eval()
        loss = 0
        pred_correct = 0
        with torch.no_grad():
            for d in tqdm(data_loader):
                input_ids = d["input_ids"]
                attention_mask = d["attention_mask"]
                targets = d["targets"]

                output = model(input_ids=input_ids, attention_mask=attention_mask)
                loss += settings['batch_size'] * loss_function(output, targets).item()
                pred = output.argmax(dim=1, keepdim=True)
                pred_correct += pred.eq(targets.view_as(pred)).sum().item()
            loss /= len(data_loader.dataset)
            acc = pred_correct / len(data_loader.dataset)
        return float(loss), float(acc)

    x_test, y_test = read_data(test_data, settings['nr_test_samples'])
    test_set = ReviewsDataset(x_test, y_test, settings['max_text_length'], settings['base_model'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=settings['batch_size'], num_workers=4)

    x_train, y_train = read_data(train_data, settings['nr_training_samples'])
    train_set = ReviewsDataset(x_train, y_train, settings['max_text_length'], settings['base_model'])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=settings['batch_size'], num_workers=4)

    try:
        test_loss, test_acc = evaluate(test_loader)
        print("Test acc: {}".format(test_acc))
        train_loss, train_acc = evaluate(train_loader)
        print("Train acc: {}".format(train_acc))

    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise

    report = {
        "classification_report": 'unevaluated',
        "training_loss": train_loss,
        "training_accuracy": train_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report


if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise e

    helper = PytorchHelper()
    model = SentimentClassifier(settings['base_model'])
    model.load_state_dict(np_to_weights(helper.load_model(sys.argv[1])))

    loss_fn = torch.nn.CrossEntropyLoss()
    report = validate(model, loss_fn, '../data/test.csv', '../data/train.csv', settings)

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))

