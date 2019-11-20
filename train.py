# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import glob
import numpy as np
import os
import io
import requests
from torch.autograd import Variable


if __name__ == '__main__':
    # *********************************************************Load the data***************************************************

    # data_paths 
    data_dir = 'flowers'

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'



    # Data transforms
    data_transforms = {
        'train': transforms.Compose([transforms.RandomRotation(25),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
        'valid' : transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
        'test' : transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    }
    train_transforms = data_transforms['train']
    valid_transforms = data_transforms['valid']
    test_transforms =data_transforms['test']



    # Image Datasets
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir,transform=train_transforms),
        'valid': datasets.ImageFolder(valid_dir,transform=valid_transforms),
        'test' : datasets.ImageFolder(test_dir,transform=test_transforms)
    }
    train_dataset =image_datasets['train']
    valid_dataset =image_datasets['valid']
    test_dataset = image_datasets['test']



    # Data loaders 
    data_loaders = {
        'train' : torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True),
        'valid': torch.utils.data.DataLoader(valid_dataset,batch_size=64),
        'test' : torch.utils.data.DataLoader(test_dataset,batch_size=64)    
    }
    train_loader = data_loaders['train']
    valid_loader = data_loaders['valid']
    test_loader  = data_loaders['test']


    # *********************************************************Label Mapping***************************************************

    import json
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print("training done")


    # ******************************************Building and training the classifier*******************************************


    # ******************************************Building the classifier*******************************************
    # selecting "densenet121" as the pretrained model as the number of inputs is less 1024 but contain more hidden layers

    n_inputs = 1024 ## will get from the pretrained model
    n_outputs = 102 # as defined in the dataset total of 102 different flower types

    # Defining the model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): # Freeze parameters so we don't backprop through them
        param.requires_grad = False


    # Defining our classifier for the pretrained model
    Classifier = nn.Sequential(nn.Linear(n_inputs,512),
                                     nn.ReLU(),
                                     nn.Dropout(0.3),                         
                                     nn.Linear(512, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.4),
                                     nn.Linear(256, n_outputs),
                                      nn.LogSoftmax(dim=1))

    model.classifier = Classifier

    criterion = nn.NLLLoss() # Defining the loss as NLLLoss

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) 


    # Move input and label tensors to the GPU and also the model
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

    model = model.to(device)

    print("training done")

    # ******************************************Training the classifier*******************************************


    def train_the_model(model,criterion,optimizer,train_loader,valid_loader,epochs):
        steps = 0
        running_loss = 0
        print_every = 5
        train_losses, valid_losses = [], []
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    train_losses.append(running_loss/print_every)
                    valid_losses.append(valid_loss/len(valid_loader))                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(valid_loader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(valid_loader):.3f}")
                    running_loss = 0
                    model.train()

        return model
    epochs = 9
    model = train_the_model(model,criterion,optimizer,train_loader,valid_loader,epochs)


    # To view the performance in a graph
    # plt.plot(train_losses, label='Training loss')
    # plt.plot(valid_losses, label='Validation loss')
    # plt.legend(frameon=False)


    # ******************************************Testing the classifier*******************************************

    def testing_model(model,test_loader,criterion):
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Test loss: {test_loss/len(test_loader):.3f}.. "
                    f"Test accuracy: {100*(accuracy/len(test_loader)):.3f}")
        return test_loss/len(test_loader),100*(accuracy/len(test_loader))


    testing_model(model,test_loader,criterion)





    # ******************************************Saving the Checkpoint*******************************************


    model.class_to_idx = train_dataset.class_to_idx
    class_names = train_dataset.classes
    saved_model = {'input_size':n_inputs,
                   'output_size':n_outputs,
                   'epochs': 5,
                   'batch_size':64,
                   'model' : models.densenet121(pretrained=True),
                   'Classifier': Classifier,
                   'optimizer' : optimizer.state_dict(),
                   'state_dict':model.state_dict(),
                   'class_to_idx': model.class_to_idx,
                   'class_names' : class_names
    }
    torch.save(saved_model,"saved_model.pth")