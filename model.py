import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class IntentModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size

    # task1: add necessary class variables as you wish.

    
    # task2: initilize the dropout and classify layers
    self.dropout = nn.Dropout(p=args.drop_rate)
    self.classify = Classifier(args, self.target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained('bert-base-uncased')
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  
    # transformer_check

    # setup optimizer and scheduler
    self.optimizer = AdamW(self.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=1000, num_training_steps=args.n_epochs*17)


  def forward(self, inputs, targets):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    # task 1: feed the input to the encoder
    outputs = self.encoder(**inputs)

    # task 2: take the last_hidden_state's <CLS> token as output of the encoder
    outputs = outputs['last_hidden_state'][:, 0, :]
    outputs = self.dropout(outputs)

    # task 3: feed the output of the dropout layer to the Classifier
    logits = self.classify(outputs)

    return logits
  
class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(IntentModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)
    
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model

class SupConModel(IntentModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
    self.head = nn.Linear(args.embed_dim, feat_dim)
 
  def forward(self, inputs, targets):

    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    # task 1: feeding the input to the encoder
    outputs = self.encoder(**inputs)

    # task 2: take last hidden state's cls token of encoder and feed to drop out
    outputs = outputs['last_hidden_state'][:, 0, :]
    outputs = self.dropout(outputs)
    # print(f"outputs: {outputs.size()}")
    # task 3: normalize output from dropout and feed to linear head
    outputs = self.head(outputs)
    # print(f"head: {outputs.size()}")
    outputs = F.normalize(outputs, dim=1)
    # print(f"normalize: {outputs.size()}")

    return outputs
