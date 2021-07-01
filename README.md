# Early Stopping by Gradient Disparity

Code for paper "Disparity Between Batches as a Signal for Early Stopping" 

## Requirements
We conducted experiments under:

    python 3.5.2
    torch 1.4.0
    torchvision 0.5.0
    cuda 10.2
    jupyter-notebook 6.0.3
    ipython 7.9.0
    1 Nvidia Titan X Maxwell GPU
Datasets:

    MNIST, CIFAR-10 and CIFAR-100 will be automatically downloaded by running the script.
    MRNet: In order to download this dataset, register here https://stanfordmlgroup.github.io/competitions/mrnet/.

## Description of files
datasets.py: the code to get data loader for MNIST, CIFAR-10 and CIFAR-100 datasets for a given batch size, training set size and level of label noise.

dataloader_mrnet.py: the code to get the data loader for MRNet dataset.

models.py: the code for neural network configurations that are used and two parameter initialization techniques.

model_mrnet.py: the code for the configuration used for MRNet dataset.

experiments.py: the code to train the models and compute gradient disparity in each epoch.

results.ipynb: the code to plot figures after the execution of experiments.py is finished.

experiments_mrnet.ipynb: the code to train the model for the MRNet dataset.

## Example 
To train a ResNet-18 on 12.8 k points of the CIFAR-10 dataset with 0 percent label noise level, batch size=128, for 20 epochs run the following command:

```
python3 experiments.py --dataset cifar10 --numsamples 12800 --batchsize 128 --corruptprob 0 --numepochs 20 --model resnet18 --filename <filename.data>
```

The results will be saved in <filename.data>.
To plot the figures of each experiment run results.ipynb file while reading the <filename.data> file of your choice.
