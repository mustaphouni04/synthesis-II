from transformers import MarianMTModel, MarianTokenizer
import torch as th
import learn2learn as l2l
from torch import nn, optim

model_name = "Helsinki-NLP/opus-mt-en-es"
tokenizer = MarianTokenizer.from_pretrained(model_name)
base_model = MarianMTModel.from_pretrained(model_name)

class MarianMAMLWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                                                                                                    )
        return outputs.loss

marianNMT = MarianMAMLWrapper(base_model)

maml = l2l.algorithms.MAML(marianNMT, lr=1e-4, first_order=False)
