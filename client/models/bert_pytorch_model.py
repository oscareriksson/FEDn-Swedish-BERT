from torch import nn
from transformers import AutoModel


class SentimentClassifier(nn.Module):
    def __init__(self, base_model, freeze_bert=True):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(base_model)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        hidden_state, _ = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=False)

        pooled_output = hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.fc(output)
