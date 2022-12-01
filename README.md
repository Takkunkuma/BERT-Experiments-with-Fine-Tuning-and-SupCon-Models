This respository is refered and modified from this repository of the [paper](https://arxiv.org/abs/2109.03079).

# PA 4: Sentence Classification/Clustering through Supervised, Contrastive Unsupervised and Contrastive Supervised networks. 


## Contributors
Andy (Jheng-Ying) Lin, Lucas Lee, Tatsuo Kumamoto


## Task
This program is intended to classify or cluster for sentences on alexa (voice assistant) and to their category/cluster of category. This task is achieved through using BERT as the encoder and using the embedding of last hidden state's CLS token. There are then 3 tasks associated with this, namely default, custom, and supcon, where each corresponds to types of models used to cluster/classify sentences.

## How to run
`python main.py --task`. The experiment can be run based on by specifying the task specified. If no task file is specified, then the default task, which is just classifying the sentences is run based on the default parameters.


## Usage
* There are many optional arguments that are available, where hyper parameters may be set.
* Namely when running the contrastive learning task,`--SimCLR` is used to define whether to use supervised(False) or unsupervised(True) loss.
* Please examine `arguments.py` for a complete list and how they can be used. T

## Files
- `main.py`: Main driver class
- `loss.py`: Calculating loss for SupCon and SimClr
- `arguments.py`: Includes the arguments possible to run the program with specific hyper parameters
- `model.py`: Definition of models for each task
- `produced_plot_*`: UMAP of clustering of based on model type. 