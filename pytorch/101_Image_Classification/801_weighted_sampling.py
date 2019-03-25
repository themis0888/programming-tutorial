# -*- coding: utf-8 -*-
"""
CUDA_VISIBLE_DEVICES=2 python -i 801_weighted_sampling.py \
--weighted=True \
--retrain True 


Transfer Learning Tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import time
import os, datetime
import copy
import pdb
module = __import__('201_vgg')
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, dest='data_path', default='/home/siit/navi/data/input_data/CLS-LOC-200/')
parser.add_argument('--save_path', type=str, dest='save_path', default='/home/siit/navi/data/meta_data/mnist_png/')
parser.add_argument('--checkpoint', type=str, dest='checkpoint', default='./checkpoint')
parser.add_argument('--log_path', type=str, dest='log_path', default='./log')
parser.add_argument('--load_date', type=str, dest='load_date', default='')

parser.add_argument('--n_classes', type=int, dest='n_classes', default=34)
parser.add_argument('--batch_size', type=int, dest='batch_size', default=8)
parser.add_argument('--step_size', type=int, dest='step_size', default=500)
parser.add_argument('--alpha', type=int, dest='alpha', default=0.5)
parser.add_argument('--save_freq', type=int, dest='save_freq', default=5)
parser.add_argument('--record_freq', type=int, dest='record_freq', default=100)

parser.add_argument('--path_label', type=bool, dest='path_label', default=False)
parser.add_argument('--weighted', type=bool, dest='weighted', default=False)
parser.add_argument('--retrain', type=bool, dest='retrain', default=False)
parser.add_argument('--iter', type=int, dest='iter', default=1)
config, unparsed = parser.parse_known_args() 

plt.ion()   # interactive mode


if not os.path.exists(config.checkpoint): os.makedirs(config.checkpoint)
if not os.path.exists(config.log_path): os.makedirs(config.log_path)

writer = SummaryWriter()

data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

batch_size = config.batch_size
step_size = config.step_size
data_dir = config.data_path
image_datasets = datasets.ImageFolder(os.path.join(data_dir),
                                          data_transforms)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size)

dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
num_class = len(class_names)
num_sample = batch_size * step_size
alpha = config.alpha
exp_time = datetime.datetime.now().strftime('%y%m%d-%H')
load_date = config.load_date
record_freq = config.record_freq

if config.weighted: mode = 'weighted'
else: mode = 'random'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


######################################################################
# Training the model
# ------------------
def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    """
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    """
    return count


def train_model(model, criterion, optimizer, scheduler, num_epochs=2501):
    since = time.time()
    f = open(os.path.join(config.log_path, exp_time + mode + '_log.txt'), 'w')

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    counter = 0
    step = 0
    train_step = 0
    
    wrong_list = [1 for i in range(num_class)]

    if not os.path.exists('imagenet_statistics.npy'):
        data_statistics = make_weights_for_balanced_classes(image_datasets, num_class)
        np.save('imagenet_statistics.npy', data_statistics)
    data_statistics = np.load('imagenet_statistics.npy')
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # label_list = [0 for i in range(num_class)]
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            
            label_list = [0 for i in range(num_class)]
            if phase == 'train':
                if step % 200000 == 0:
                    scheduler.step()
                model.train() 
                
                total_wrong = 0
                # for ele in wrong_list: total_wrong += ele
                weights = []
                if config.weighted:
                    weights_list = [(1-alpha) / num_class + alpha * x / sum(wrong_list) for x in wrong_list]
                    for index in range(num_class):
                        num_data = data_statistics[index]
                        weights += [weights_list[index] / num_data] * num_data
                else: 
                    weights_list = [1 / num_class for x in range(len(wrong_list))]
                    weights = [1] * len(image_datasets)

                print(weights[:3])
                sampler = torch.utils.data.sampler.WeightedRandomSampler(  # torch.tensor(weights).type('torch.DoubleTensor')
                    weights, batch_size * step_size) # This line has been changed
                # sampler = torch.utils.data.sampler.RandomSampler(image_datasets)
                dataloaders = torch.utils.data.DataLoader(image_datasets, 
                    batch_size=batch_size, sampler = sampler) 
                # Set model to training mode
                

            else:
                model.eval()
                wrong_list = [1 for i in range(num_class)]
                
                dataloaders = torch.utils.data.DataLoader(
                    image_datasets, batch_size=batch_size, shuffle=True)  
                # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            record_loss = 0.0
            record_acc = 0
            # Iterate over data.
            for counter in range(step_size):

                inputs, labels = next(iter(dataloaders))
                # print(labels)
                step += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        train_step += 1
                        loss.backward()
                        optimizer.step()
                        for i in range(batch_size):
                            label_list[labels[i]] += 1

                
                if phase == 'val':
                    for i in range(batch_size):
                        if labels[i] != preds[i]:
                            wrong_list[labels[i]] += 1
                    

                # statistics
                cur_loss = loss.item() * inputs.size(0)
                running_loss += cur_loss
                running_corrects += torch.sum(preds == labels.data)
                record_loss += cur_loss
                record_acc += torch.sum(preds == labels.data)
                
                if step % record_freq == 0:
                    print('{} Step \tLoss: {:.4f} Acc: {:.4f}'.format(
                        step, cur_loss, torch.sum(preds == labels.data)/batch_size))
                    

                    writer.add_scalars('data/loss', {mode: record_loss / record_freq / batch_size}, train_step)
                    writer.add_scalars('data/acc', {mode: record_acc / record_freq / batch_size}, train_step)
                    record_loss = 0.0
                    record_acc = 0
                     
                    for i in range(batch_size):
                        label_pred = mode + '\: ' + class_names[labels[i]] + ' \@ ' + class_names[preds[i]]
                        writer.add_image(mode + '_Image{}'.format(i), inputs[i], train_step)


            # pdb.set_trace()
            epoch_loss = running_loss / num_sample
            epoch_acc = running_corrects.double() / num_sample

            epoch_log = '{} Loss: {:.4f} Acc: {:.4f} \n'.format(
                phase, epoch_loss, epoch_acc)
            print(epoch_log)
            f.write(epoch_log)
            # pdb.set_trace()
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if epoch % config.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(config.checkpoint, exp_time + mode + '_model.pt'))
            
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def visualize_model(model, num_images=batch_size):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

######################################################################
# Finetuning the convnet


model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
    
model_ft.fc = nn.Linear(num_ftrs, num_class)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.3)

######################################################################
# Train and evaluate
if config.retrain:
    model_ft.load_state_dict(torch.load(
        os.path.join(config.checkpoint, load_date + mode + '_model.pt')
    ))
    print(load_date + mode + '_model.pt loaded')

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
writer.export_scalars_to_json("./all_scalars.json")
writer.close()

visualize_model(model_ft)


model_conv = torchvision.models.resnet50(pretrained=False)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, num_class)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# On CPU this will take about half the time compared to previous scenario.
# This is expected as gradients don't need to be computed for most of the
# network. However, forward does need to be computed.
#

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)

######################################################################
#

visualize_model(model_conv)

plt.ioff()
plt.show()
