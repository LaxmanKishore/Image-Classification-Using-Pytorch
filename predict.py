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

import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # add the parameters image_path 
    parser.add_argument('image_path',help='image path')
    
    # parse the arguments
    args = parser.parse_args()
    image_path = args.image_path

    
    # ******************************************Loading the Checkpoint*******************************************

    device = torch.device("cuda")
    loaded_model = torch.load('saved_model.pth')
    def load_model(file_path,map_location='cpu'):

        loaded_model = torch.load(file_path)
        model1 = loaded_model['model']
        model1.classifier = loaded_model['Classifier']
        model1.load_state_dict(loaded_model['state_dict'],strict=False)
        model1.class_to_idx = loaded_model['class_to_idx']
        optimizer = loaded_model['optimizer']
        epochs = loaded_model['epochs']
        class_names = loaded_model['class_names']

        for param in model1.parameters():
            param.requires_grad = False

        return model1,loaded_model['class_to_idx'],class_names

    model,class_to_idx,class_names = load_model('saved_model.pth')

    model.to(device)


    #******************************************Processing the input image*******************************************

    def process_image(image):

        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # TODO: Process a PIL image for use in a PyTorch model
        transform = transforms.Compose([
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
                     ])

        return transform(image)



    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        return ax

    # ******************************************Predicting the image*******************************************
    def predict(image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        model.eval()
        image = Image.open(image_path)
        img =process_image(image)

        img = np.expand_dims(img,0) # 2D to 1D

        img = torch.from_numpy(img)

        inputs = Variable(img).to(device)
        log_ps = model.forward(inputs)

        ps = F.softmax(log_ps,dim=1)
        topk = ps.cpu().topk(topk)

        return (i.data.numpy().squeeze().tolist() for i in topk)


    #image_path = "assets/Capture_test_flower.JPG"
    probs, classes = predict(image_path, model)
    print(classes)
    print(probs)

    #***************************Getting top 5 predicted names for the flower*****************************
    flower_names= [cat_to_name[class_names[i]] for i in classes]
    print(flower_names)


    #******************************************Sanity Checking*******************************************

    def view_classify(image_path,probs,classes,mapping):
        # to view image and get predicted class
        image = Image.open(image_path)

        fig,(ax1,ax2) = plt.subplots(figsize=(8,10),ncols=1,nrows=2)
        flower_name = flower_names[0]
        ax1.set_title(flower_name)
        ax1.imshow(image)
        ax1.axis('off')

        y_pos = np.arange(len(probs))
        ax2.barh(y_pos,probs,align='center')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(flower_names)
        ax2.invert_yaxis()
        ax2.set_title('Class Probability')

    view_classify(image_path,probs,classes,cat_to_name)
