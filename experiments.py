import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import copy
from torch.autograd.gradcheck import zero_gradients
import matplotlib.pyplot as plt
import math
from torch.autograd import grad
from datasets import *
from models import *
import simplejson
import argparse
import pickle

def accuracy_top5(output, target, topk=(5,)):
    """computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(args, epoch, dataset_loader, clf):
    """function for training the network
       In each epoch it trains over each training sample once
       over the mini batches and returns the average loss and accuracy
    """
    
    # set model in training mode (need this because of dropout)
    clf.train() 
    
    train_loss = 0
    train_err = 0
    correct = 0
    top5_correct = 0
    opt = optim.SGD(clf.parameters(), lr=args.lr)
    criteria = torch.nn.CrossEntropyLoss()
    
    # dataset API gives us pythonic batching 
    for batch_id, (data, label) in enumerate(dataset_loader):
        data = Variable(data).cuda()
        target = Variable(label).cuda()
        
        # forward pass, calculate loss and backprop
        opt.zero_grad()
        outs = clf(data)
        loss = criteria(outs, target)

        # taking the average over training losses
        train_loss += loss.item()
        
        # finding the accuracy
        _, pred = torch.max(outs.data, 1)
        correct += pred.eq(target.data).sum().item()
        c = pred.eq(target.data).sum().item()
        e = 100.0 - 100.0*c/args.batchsize
        top5_correct += accuracy_top5(outs, target)[-1].item()
        
        loss.backward()
        opt.step()
      
        
    train_loss /= len(dataset_loader)
    accuracy = 100. * correct / (len(dataset_loader.dataset))
    train_err = 100 - accuracy
    top5_correct /= len(dataset_loader)
    return train_loss, train_err, top5_correct

def test(args, dataset_loader, clf):
    """function for evaluating the model on the unseen data
    """
    with torch.no_grad():
        clf.eval()
        criteria = torch.nn.CrossEntropyLoss()
        
        test_loss = 0
        correct = 0
        top5_correct = 0
        
        for data, target in dataset_loader:
            data = Variable(data).cuda()
            target = Variable(target).cuda()
            
            output = clf(data)
            
            # find loss
            test_loss += criteria(output, target).item()
            
            _, pred = torch.max(output.data, 1) # get the index of the max log-probability
            correct += pred.eq(target.data).sum().item()
            
            top5_correct += accuracy_top5(output, target)[-1].item()
            
            
        test_loss /= len(dataset_loader) # loss function already averages over batch sizes
        accuracy = 100. * correct / (len(dataset_loader.dataset))
        err = 100 - accuracy
        top5_correct /= len(dataset_loader)
        return test_loss, err, top5_correct

def find_grad_dis(args, dataset_loader, clf):
    """the function for computing the gradient disparity
    this functions gives the following output:
    avg_grad_dis: the avg gradient disparity between pairs of batches of the dataset
    """
    criteria = torch.nn.CrossEntropyLoss(reduction='none') # then instead of mean 
    # we have minibatch size values for loss to perform the re-scaling on
    
    opt = optim.SGD(clf.parameters(), lr=args.lr)
    
    # set model in training mode 
    clf.train() 
    
    cnt = 0
    Grads = []
    for batch_id, (data, label) in enumerate(dataset_loader):
        if batch_id < args.s: 
            data1 = Variable(data).cuda()
            target1 = Variable(label).cuda()

            opt.zero_grad()
            output1 = clf(data1)
            loss1 = criteria(output1, target1)             
            loss1_s = loss1/torch.std(loss1)

            loss1_s = torch.mean(loss1_s)

            loss1_s.backward(retain_graph=True)
            grads1_s = []
            for param in clf.parameters():
                grads1_s.append(param.grad.view(-1))
            grads1_s = torch.cat(grads1_s)
            Grads.append(grads1_s.data.cpu().numpy())
            cnt += 1
    
    Grads = np.array(Grads)
    cnt2 = 0
    avg_grad_dis = 0
    # now compute the pairwise \ell_2 norm distance
    for i in range(cnt):
        for j in range(cnt):
            if i < j:
                grads1 = Grads[i]
                grads2 = Grads[j]
                GD = np.linalg.norm(grads1-grads2)
                avg_grad_dis += GD
                cnt2 += 1
    # to avoid division by zero here, args.s must be >= 2
    avg_grad_dis /= cnt2
    
    return avg_grad_dis
def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch Gradient Disparity Code")
    parser.add_argument("--model", type=str, choices=['alexnet', 'fc', 'VGG11','VGG13','VGG16','VGG19','resnet18','resnet34'], help="the neural network configuration")
    parser.add_argument("--init", type=str, choices=['SN', 'HN'], default= None, help="the parameter initialization")
    parser.add_argument("--scale", type=float, default=1, help="the scale to use for the configuration, for FC it is the number of hidden units")
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset choices are mnist, cifar10, anf cifar100")
    parser.add_argument("--numsamples", type=int, default=12800, help="number of training samples to train on")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size of both the training and the testing sets")
    parser.add_argument("--corruptprob", type=float, default=0.0, help="the corrupt probability of the labels of the training samples")
    parser.add_argument("--numepochs", type=int, default=500, help="maximum number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="sgd learning rate")
    parser.add_argument("--s", type=int, default=5, help="number of pairs of batches to compute average gradient disparity (s in the paper)")
    parser.add_argument("--logevery", type=int, default=10, help="frequency of printing the loss and accuracy")
    parser.add_argument("--filename", type=str, default='temp.data', help="filename to save the results to")
    args = parser.parse_args()
    # data set parameters
    if args.dataset == 'mnist':
        input_size = 28*28
        num_classes = 10
    elif args.dataset == 'cifar10':
        input_size = 32*32*3
        num_classes = 10
    elif args.dataset == 'cifar100':
        input_size = 32*32*3
        num_classes = 100
    
    # Device configuration
    device = torch.cuda.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data loader
    my_train_loader, my_test_loader, my_train_dataset = get_data_loader(args.dataset, batch_size=args.batchsize, num_samples=args.numsamples, corrupt_prob = args.corruptprob)
    
    vLosses = []
    tLosses = []
    vERRs = []
    tERRs = []
    vACCs5 = []
    tACCs5 = []
    avg_GD = []

    clf = get_model(args.model, input_size, num_classes, args.scale, args.init).cuda()
    #clf = nn.DataParallel(clf) # to run on multiple gpus
    
    
    epoch = 0
    # train the network
    while epoch < args.numepochs:
        # train the network
        t_loss, t_err, t_acc5 = train(args, epoch, my_train_loader, clf)
        tLosses.append(t_loss)
        tERRs.append(t_err)
        tACCs5.append(t_acc5)
        
        ce, err, acc5 = test(args, my_test_loader, clf)
        vLosses.append(ce)
        vERRs.append(err)
        vACCs5.append(acc5)
        
        if epoch%args.logevery == 0:
            print('epoch: ', epoch, ' loss: ', t_loss, ' acc: ',100.0 - t_err)
            print('val loss: ', ce, ' acc: ', 100.0 - err)
        
        ag = find_grad_dis(args, my_train_loader, clf)
        avg_GD.append(ag)

        epoch += 1
        torch.cuda.empty_cache()
    
    
    # to save the results in a file
    List = [vLosses, tLosses, vERRs, tERRs, vACCs5, tACCs5, avg_GD]
    with open(args.filename, 'wb') as filehandle:
        # store the data as binary data stream
        for ls in List:
            pickle.dump(ls, filehandle)    
    
    
    
    
    # to perform k-fold cross validation: comment the above lines from line 197 and uncomment the lines below
    """
    k = 5 # k in the k-fold cross validation
    vLosses = []
    tLosses = []
    dLosses = []
    vERRs = []
    tERRs = []
    dERRs = []
    vACCs5 = []
    tACCs5 = []
    dACCs5 = []
    fold_size = int(len(my_train_dataset)/k)

    Train_sets = []
    for i in range(k):
        train_set, my_train_dataset = torch.utils.data.random_split(dataset=my_train_dataset, 
                                        lengths=[fold_size,len(my_train_dataset)-fold_size])
        Train_sets.append(train_set)
    for n in range(k):
        print('### %s th fold ###' % (n))
        # set fold n for val and the rest for train
        train_split_set = torch.utils.data.ConcatDataset([Train_sets[i] for i in range(k) if i != n])
        dev_split_set = Train_sets[n]

        train_split = torch.utils.data.DataLoader(dataset=train_split_set, 
                                           batch_size=batch_size, 
                                           shuffle=True) # this shuffle is for iterations within each fold
        dev_split = torch.utils.data.DataLoader(dataset=dev_split_set, 
                                           batch_size=batch_size, 
                                           shuffle=True)
        vLs = []
        tLs = []
        dLs = []
        vEs = []
        tEs = []
        dEs = []
        vA5 = []
        tA5 = []
        dA5 = []

        clf = get_model(args.model, input_size, num_classes, args.scale, args.init).cuda()
        #clf = nn.DataParallel(clf) # to run on multiple gpus
        
        while epoch < num_epochs:
        
            # train k-1 out of k
            t_loss, t_err, t_acc_5 = train(args, epoch, train_split, clf)
            tLs.append(t_loss)
            tEs.append(t_err)
            tA5.append(t_acc_5)

            # evaluate on the rest
            dev_loss, dev_err, dev_acc_5 = test(args, dev_split, clf)
            dLs.append(dev_loss)
            dEs.append(dev_err)
            dA5.append(dev_acc_5)
            
            # evaluate on unseen data (held out data)
            ce, err, acc_5 = test(test_loader)
            vLs.append(ce)
            vEs.append(err)
            vA5.append(acc_5)
        
        
            if epoch%args.logevery == 0:
                print('epoch: ', epoch, ' loss: ', t_loss, ' acc: ',100.0 - t_err, 'top5 acc: ', t_err_5, 'dev loss: ', dev_loss, 'dev acc: ',100- dev_err, 'top 5 acc: ', dev_err_5)
                print('val loss: ', ce, ' acc: ', 100.0 - err, 'top5 acc: ', err_5)
        
            epoch += 1
            torch.cuda.empty_cache()
        
        
        vLosses.append(vLs)
        tLosses.append(tLs)
        dLosses.append(dLs)
        vERRs.append(vEs)
        tERRs.append(tEs)
        dERRs.append(dEs)
        vACCs5.append(vA5)
        tACCs5.append(tA5)
        dACCs5.append(dA5)
        
    # to save the results in a file
    List = [vLosses, tLosses, dLosses, vERRs, tERRs, dERRs, vACCs5, tACCs5, dACCs5]
    
    with open(args.filename, 'wb') as filehandle:
        # store the data as binary data stream
        for ls in List:
            pickle.dump(ls, filehandle)    
        
    """
    
    
if __name__ == "__main__":
    main()    
    
    
    
    
    
    