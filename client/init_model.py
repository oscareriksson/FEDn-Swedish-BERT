import yaml

from fedn.utils.pytorchhelper import PytorchHelper
from models.bert_pytorch_model import SentimentClassifier
from util.transformations import weights_to_np


if __name__ == '__main__':

	with open('settings.yaml', 'r') as fh:
		try:
			settings = dict(yaml.safe_load(fh))
		except yaml.YAMLError as e:
			raise e

	model = SentimentClassifier(settings['base_model'])
	outfile_name = "../initial_model/initial_model.npz"
	helper = PytorchHelper()
	helper.save_model(weights_to_np(model.state_dict()), outfile_name)
