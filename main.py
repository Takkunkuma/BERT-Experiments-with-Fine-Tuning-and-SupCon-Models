import os, sys, pdb
import numpy as np
import random
import torch
import umap
import umap.plot
print('=====UMAP IMPORTED=====')
import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel
from torch import nn

device='cuda'

def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    # REVISE
    train_dataloader = get_dataloader(args, datasets['train'], split='train')

    # task2: setup model's optimizer_scheduler if you have
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
        run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()

    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    
    val_accs = []
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            logits = model(inputs, labels)
            loss = criterion(logits, labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()
            # run validation every n batches
            if step % args.eval_every == 0:
                val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
                # early stopping if acc decrease
                if len(val_accs) > 1 and val_acc < val_accs[-1]:
                    print('early stopping')
                    return
                val_accs.append(val_acc)
                model.train()
        print('epoch', epoch_count, '| losses:', losses)
        

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))

    return acc/len(datasets[split])

def supcon_train(args, model, datasets, tokenizer):
    if args.CrossCluster:
        criterion = nn.CrossEntropyLoss()
    else:
        from loss import SupConLoss
        criterion = SupConLoss(temperature=args.temperature)

    # task1: load training split of the dataset
    train_dataloader = get_dataloader(args, datasets['train'], split='train')
    
    # task2: setup optimizer_scheduler in your model
    # already done in model.py

    # task3: write a training loop for SupConLoss function 
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs, labels = prepare_inputs(batch, model)
            
            if args.CrossCluster == False: 
                embeddings_1 = model(inputs, labels) # should be 2n x d
                embeddings_2 = model(inputs, labels)
                embeddings = torch.cat([embeddings_1.unsqueeze(1), embeddings_2.unsqueeze(1)], dim=1)
                # use SimClR or SupCon based on arguments
                if args.SimCLR:
                    #print("=====Running SimCLR Loss=====")
                    loss = criterion(embeddings)
                else:
                    #print("=====Running SupCon Loss=====")
                    loss = criterion(embeddings, labels)
            else:
                #print("=====Running CrossEntropy Loss=====")
                embeddings = model(inputs, labels)
                loss = criterion(embeddings, labels)

            losses += loss.item()
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            model.scheduler.step()
        
        # run validation every epoch
        run_eval(args, model, datasets, tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses)

    data = []
    classes = []
    for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
        inputs, labels = prepare_inputs(batch, model)
        idx = [labels < 10][0].detach().to('cpu').numpy()
        #print("=====PREPARED INPUTS=====")
        embeds = model(inputs, labels)
        for l,e,i in zip(labels, embeds, idx):
            #print("=====ZIPPING RIGHT NOW=====")
            if (i == True):
                data.append(e.detach().to('cpu').numpy())
                classes.append(l.detach().to('cpu').numpy())
                
                
    embeddings = np.stack(data) #2D of total N X feat_dim
    labels = np.stack(classes)

    mapper = umap.UMAP().fit(embeddings)
    image = umap.plot.points(mapper, labels=labels)
    figure = image.get_figure()
    if args.CrossCluster:
        figure.savefig("produced_plot_crossentropy")
    elif args.SimCLR:
        figure.savefig("produced_plot_SimCLR")
    else:
        figure.savefig("produced_plot_supcon")
            

if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
 
  if args.task == 'baseline':
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    model = CustomModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets, tokenizer)
   
