#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#from comet_ml import Experiment
#experiment = Experiment()


# In[ ]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image, ImageFile
from pickle import load, dump
import cv2
import time
import argparse
import subprocess
from sklearn.metrics import precision_score, recall_score, f1_score
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


class BasicConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.activation1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.activation2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(192)
        self.activation3 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation3(x)
        x = self.maxpool2(x)
        return x


# In[ ]:


class InceptionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.convA1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormA1 = nn.BatchNorm2d(64)
        self.activationA1 = nn.ReLU(inplace=True)
        
        self.convB1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormB1 = nn.BatchNorm2d(64)
        self.activationB1 = nn.ReLU(inplace=True)
        self.convB2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batchnormB2 = nn.BatchNorm2d(64)
        self.activationB2 = nn.ReLU(inplace=True)
        
        self.convC1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormC1 = nn.BatchNorm2d(64)
        self.activationC1 = nn.ReLU(inplace=True)
        self.convC2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.batchnormC2 = nn.BatchNorm2d(96)
        self.activationC2 = nn.ReLU(inplace=True)
        self.convC3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.batchnormC3 = nn.BatchNorm2d(96)
        self.activationC3 = nn.ReLU(inplace=True)
        
        self.avgpoolD1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.convD1 = nn.Conv2d(192, 32, kernel_size=1, stride=1, padding=0)
        self.batchnormD1 = nn.BatchNorm2d(32)
        self.activationD1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out1 = self.convA1(x)
        out1 = self.batchnormA1(out1)
        out1 = self.activationA1(out1)
        
        out2 = self.convB1(x)
        out2 = self.batchnormB1(out2)
        out2 = self.activationB1(out2)
        out2 = self.convB2(out2)
        out2 = self.batchnormB2(out2)
        out2 = self.activationB2(out2)
        
        out3 = self.convC1(x)
        out3 = self.batchnormC1(out3)
        out3 = self.activationC1(out3)
        out3 = self.convC2(out3)
        out3 = self.batchnormC2(out3)
        out3 = self.activationC2(out3)
        out3 = self.convC3(out3)
        out3 = self.batchnormC3(out3)
        out3 = self.activationC3(out3)
        
        out4 = self.avgpoolD1(x)
        out4 = self.convD1(out4)
        out4 = self.batchnormD1(out4)
        out4 = self.activationD1(out4)
        
        out = [out1, out2, out3, out4]
        return torch.cat(out, 1)


# In[ ]:


class InceptionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.convA1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormA1 = nn.BatchNorm2d(64)
        self.activationA1 = nn.ReLU(inplace=True)
        
        self.convB1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormB1 = nn.BatchNorm2d(64)
        self.activationB1 = nn.ReLU(inplace=True)
        self.convB2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.batchnormB2 = nn.BatchNorm2d(96)
        self.activationB2 = nn.ReLU(inplace=True)
        
        self.convC1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormC1 = nn.BatchNorm2d(64)
        self.activationC1 = nn.ReLU(inplace=True)
        self.convC2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.batchnormC2 = nn.BatchNorm2d(96)
        self.activationC2 = nn.ReLU(inplace=True)
        self.convC3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.batchnormC3 = nn.BatchNorm2d(96)
        self.activationC3 = nn.ReLU(inplace=True)
        
        self.avgpoolD1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.convD1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormD1 = nn.BatchNorm2d(64)
        self.activationD1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out1 = self.convA1(x)
        out1 = self.batchnormA1(out1)
        out1 = self.activationA1(out1)
        
        out2 = self.convB1(x)
        out2 = self.batchnormB1(out2)
        out2 = self.activationB1(out2)
        out2 = self.convB2(out2)
        out2 = self.batchnormB2(out2)
        out2 = self.activationB2(out2)
        
        out3 = self.convC1(x)
        out3 = self.batchnormC1(out3)
        out3 = self.activationC1(out3)
        out3 = self.convC2(out3)
        out3 = self.batchnormC2(out3)
        out3 = self.activationC2(out3)
        out3 = self.convC3(out3)
        out3 = self.batchnormC3(out3)
        out3 = self.activationC3(out3)
        
        out4 = self.avgpoolD1(x)
        out4 = self.convD1(out4)
        out4 = self.batchnormD1(out4)
        out4 = self.activationD1(out4)
        
        out = [out1, out2, out3, out4]
        return torch.cat(out, 1)


# In[ ]:


class InceptionC(nn.Module):
    def __init__(self):
        super().__init__()
        self.convA1 = nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0)
        self.batchnormA1 = nn.BatchNorm2d(64)
        self.activationA1 = nn.ReLU(inplace=True)
        self.convA2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.batchnormA2 = nn.BatchNorm2d(96)
        self.activationA2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.convA1(x)
        out = self.batchnormA1(out)
        out = self.activationA1(out)
        out = self.convA2(out)
        out = self.batchnormA2(out)
        out = self.activationA2(out)
        return out


# In[ ]:


class ECO2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_conv = BasicConv()
        self.inceptionA = InceptionA()
        self.inceptionB = InceptionB()
        self.inceptionC = InceptionC()
        
    def forward(self, x):
        x = self.basic_conv(x)
        x = self.inceptionA(x)
        x = self.inceptionB(x)
        x = self.inceptionC(x)
        return x


# In[ ]:


class ResNet3dA(nn.Module):
    def __init__(self):
        super().__init__()
        self.convA1 = nn.Conv3d(96, 128, kernel_size=3, stride=1, padding=1)
        
        self.batchnormB1 = nn.BatchNorm3d(128)
        self.activationB1 = nn.ReLU(inplace=True)
        self.convB2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batchnormB2 = nn.BatchNorm3d(128)
        self.activationB2 = nn.ReLU(inplace=True)
        self.convB3 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        
        self.batchnormC1 = nn.BatchNorm3d(128)
        self.activationC1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.convA1(x)
        
        out = self.batchnormB1(residual)
        out = self.activationB1(out)
        out = self.convB2(out)
        out = self.batchnormB2(out)
        out = self.activationB2(out)
        out = self.convB3(out)
        
        out += residual
        
        out = self.batchnormC1(out)
        out = self.activationC1(out)
        
        return out


# In[ ]:


class ResNet3dB(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convA1 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.convB1 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.batchnormB1 = nn.BatchNorm3d(256)
        self.activationB1 = nn.ReLU(inplace=True)
        self.convB2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.batchnormC1 = nn.BatchNorm3d(256)
        self.activationC1 = nn.ReLU(inplace=True)
        self.convC2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batchnormC2 = nn.BatchNorm3d(256)
        self.activationC2 = nn.ReLU(inplace=True)
        self.convC3 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.batchnormD1 = nn.BatchNorm3d(256)
        self.activationD1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.convA1(x)
        
        out = self.convB1(x)
        out = self.batchnormB1(out)
        out = self.activationB1(out)
        out = self.convB2(out)
        
        out += residual
        residual = out
        
        out = self.batchnormC1(out)
        out = self.activationC1(out)
        out = self.convC2(out)
        out = self.batchnormC2(out)
        out = self.activationC2(out)
        out = self.convC3(out)
        
        out += residual
        
        out = self.batchnormD1(out)
        out = self.activationD1(out)
        
        return out


# In[ ]:


class ResNet3dC(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convA1 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.convB1 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.batchnormB1 = nn.BatchNorm3d(512)
        self.activationB1 = nn.ReLU(inplace=True)
        self.convB2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.batchnormC1 = nn.BatchNorm3d(512)
        self.activationC1 = nn.ReLU(inplace=True)
        self.convC2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batchnormC2 = nn.BatchNorm3d(512)
        self.activationC2 = nn.ReLU(inplace=True)
        self.convC3 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.batchnormD1 = nn.BatchNorm3d(512)
        self.activationD1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = self.convA1(x)
        
        out = self.convB1(x)
        out = self.batchnormB1(out)
        out = self.activationB1(out)
        out = self.convB2(out)
        
        out += residual
        residual = out
        
        out = self.batchnormC1(out)
        out = self.activationC1(out)
        out = self.convC2(out)
        out = self.batchnormC2(out)
        out = self.activationC2(out)
        out = self.convC3(out)
        
        out += residual
        
        out = self.batchnormD1(out)
        out = self.activationD1(out)
        
        return out


# In[ ]:


class ECO3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet3dA = ResNet3dA()
        self.resnet3dB = ResNet3dB()
        self.resnet3dC = ResNet3dC()
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.resnet3dA(x)
        x = self.resnet3dB(x)
        x = self.resnet3dC(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x


# In[ ]:


class ECO(nn.Module):
    def __init__(self, num_label):
        super().__init__()
        self.eco2d = ECO2d()
        self.eco3d = ECO3d()
        self.linear = nn.Linear(512, num_label)
        
    def forward(self, x):
        batch, segments, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)
        x = self.eco2d(x)
        x = x.view(-1, segments, 96, 28, 28)
        x = self.eco3d(x)
        x = self.linear(x)
        return x


# In[ ]:


class VideoTransform():
    def __init__(self, resize, crop_size):
        self.transform = transforms.Compose([
            GroupResize(int(resize)),
            GroupCenterCrop(crop_size),
            GroupToTensor(),
            Stack()
        ])
        
    def  __call__(self, images):
        return self.transform(images)


# In[ ]:


class GroupResize():
    def __init__(self, resize):
        self.transform = transforms.Resize(resize)
    
    def __call__(self, images):
        return [self.transform(image) for image in images]


# In[ ]:


class GroupCenterCrop():
    def __init__(self, crop_size):
        self.transform = transforms.CenterCrop(crop_size)
    
    def __call__(self, images):
        return [self.transform(image) for image in images]


# In[ ]:


class GroupToTensor():
    def __init__(self):
        self.transform = transforms.ToTensor()
    
    def __call__(self, images):
        return [self.transform(image) for image in images]


# In[ ]:


class Stack():
    def __call__(self, images):
        return torch.cat([image.unsqueeze(dim=0) for image in images], dim=0)


# In[ ]:


class VideoDataset(Dataset):
    def __init__(self, videos, label2id, num_segments, transform, image_format='image_{:05d}.jpg'):
        self.videos = videos
        self.label2id = label2id
        self.num_segments = num_segments
        self.transform = transform
        self.image_format = image_format
        
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        path = self.videos[index]
        files = os.listdir(path)
        tick = len(files) / float(self.num_segments)
        indices = np.array([int(tick / 2 + tick * i) for i in range(self.num_segments)]) + 1
        images = []
        for idx in indices:
            file = os.path.join(path, self.image_format.format(idx))
            image = Image.open(file).convert('RGB')
            images += [image]
        label = (path[:-1] if path.endswith('/') else path).split('/')[-2]
        label_id = self.label2id[label]
        images = self.transform(images)
        return images, label, label_id


# In[ ]:


class Util:
    @staticmethod
    def loadImages(batch_size, folder_path, size):
        imgs = ImageFolder(folder_path, transform=transforms.Compose([
            transforms.Resize(int(size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size),
            #transforms.RandomRotation(degrees=30),
            #transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0.5),
            transforms.ToTensor()
        ]))
        return DataLoader(imgs, batch_size=batch_size, shuffle=True, drop_last=True)
    
    @staticmethod
    def showImages(dataloader):
        get_ipython().run_line_magic('matplotlib', 'inline')
        import matplotlib.pyplot as plt
        
        PIL = transforms.ToPILImage()
        ToTensor = transforms.ToTensor()

        for images in dataloader:
            for image in images[0]:
                img = PIL(image)
                fig = plt.figure(dpi=200)
                ax = fig.add_subplot(1, 1, 1) # (row, col, num)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(img)
                #plt.gray()
                plt.show()


# In[ ]:


class Solver:
    def __init__(self, use_cpu, epochs, lr, batch_size, resize, crop_size, num_segments, videos, weights_dir):
        use_cuda = torch.cuda.is_available() if not use_cpu else False
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        torch.backends.cudnn.benchmark = True
        print(f'Use Device: {self.device}')
        
        self.num_epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.movie_resize = resize
        self.crop_size = crop_size
        self.weights_dir = weights_dir
        
        self.labels = [f for f in os.listdir(videos) if os.path.isdir(os.path.join(videos, f))]
        self.label2id = {v:i for i, v in enumerate(self.labels)}
        
        self.videos = []
        for label in self.labels:
            path = os.path.join(videos, label)
            for file in os.listdir(path):
                file = os.path.join(path, file)
                if os.path.isdir(file):
                    self.videos += [file]
        
        self.num_label = len(self.labels)
        self.num_segments = num_segments
        
        self.dataset = VideoDataset(self.videos, self.label2id, num_segments,
                                    VideoTransform(resize, crop_size))
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        self.net = ECO(self.num_label).to(self.device)
        self.state_loaded = False

        self.net.apply(self.weights_init)
        
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        self.epoch = 0
    
    def weights_init(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d or type(module) == nn.ConvTranspose2d:
            nn.init.kaiming_normal_(module.weight)
            module.bias.data.fill_(0)
            
    def save_labeldata(self):
        with open(os.path.join('.', f'labels.pkl'), 'wb') as f:
            dump(self.labels, f)
        with open(os.path.join('.', f'label2id.pkl'), 'wb') as f:
            dump(self.label2id, f)
        print('Saved labeldata.')
            
    def load_labeldata(self):
        if os.path.exists('labels.pkl') and os.path.exists('label2id.pkl'):
            with open(os.path.join('.', 'labels.pkl'), 'rb') as f:
                print('Load labels.')
                self.labels = load(f)
            with open(os.path.join('.', 'label2id.pkl'), 'rb') as f:
                print('Load label2id.')
                self.label2id = load(f)
        self.num_label = len(self.labels)
            
    def save_state(self, epoch):
        self.net.cpu()
        torch.save(self.net.state_dict(), os.path.join(self.weights_dir, f'weight.{epoch}.pth'))
        self.net.to(self.device)
        
    def load_state(self):
        if os.path.exists('weight.pth'):
            self.net.load_state_dict(torch.load('weight.pth', map_location=self.device))
            self.state_loaded = True
            print('Loaded network state.')
    
    def save_resume(self):
        with open(os.path.join('.', f'resume.pkl'), 'wb') as f:
            dump(self, f)
    
    def load_resume(self):
        if os.path.exists('resume.pkl'):
            with open(os.path.join('.', 'resume.pkl'), 'rb') as f:
                print('Load resume.')
                return load(f)
        else:
            return self
    
    def train(self, resume=True):
        self.net.train()
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(1, self.num_epochs + 1):
            if epoch < self.epoch:
                continue
            self.epoch = epoch
            
            epoch_loss = 0.0
            
            for iters, (images, label, label_id) in enumerate(tqdm(self.dataloader)):
                iters += 1
                
                images = images.to(self.device)
                label_id = label_id.to(self.device)
                
                out = self.net(images)
                loss = criterion(out, label_id)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss
                #experiment.log_metric('Loss', loss)
                
            self.save_state(epoch)
            print(f'Epoch[{self.epoch}] Loss[{epoch_loss}]')
                    
            if resume:
                self.save_resume()
    
    def validate(self):
        self.net.eval()
        
        valid = []
        for iters, (images, label, label_id) in enumerate(tqdm(self.dataloader)):
            images = images.to(self.device)
            out = self.net(images)
            
            pred_label = np.argmax(out.cpu().detach().numpy(), axis=1)
            label_id = label_id.detach().numpy()
            
            precision = precision_score(label_id, pred_label, average='macro', zero_division=1)
            recall = recall_score(label_id, pred_label, average='macro', zero_division=1)
            f1 = f1_score(label_id, pred_label, average='macro', zero_division=1)
            valid += [[precision, recall, f1]]
            
        valid = np.mean(valid, axis=0)
        print(f'Precision: {valid[0]}')
        print(f'Recall: {valid[1]}')
        print(f'F1: {valid[2]}')
    
    def predict(self, file):
        self.net.eval()
        
        name, ext = os.path.splitext(file)
        if not os.path.exists(name):
            os.mkdir(name)
        command = f'ffmpeg -i \"{file}\" -vf scale=-1:256 \"{name}/image_%05d.jpg\"'
        print(command)
        subprocess.call(command, shell=True)
        print('Converted data.')
        
        files = os.listdir(name)
        tick = len(files) / float(self.num_segments)
        indices = np.array([int(tick / 2 + tick * i) for i in range(self.num_segments)]) + 1
        images = []
        for idx in indices:
            file = os.path.join(name, f'image_{idx:05d}.jpg')
            image = Image.open(file).convert('RGB')
            images += [image]
        images = VideoTransform(self.movie_resize, self.crop_size)(images).unsqueeze(0).to(self.device)
        
        pred = F.softmax(self.net(images), dim=1).cpu().detach().numpy()[0]
        indexes = np.argsort(pred)[::-1]
        print('Predicted:')
        for i in indexes:
            print(f'\t{self.labels[i]}: {pred[i]}')


# In[ ]:


def main(args):
    hyper_params = {}
    hyper_params['Videos Dir'] = args.videos
    hyper_params['Weights Dir'] = args.weights_dir
    hyper_params['Data Resize'] = args.resize
    hyper_params['Data Crop Size'] = args.crop_size
    hyper_params['Num Segments'] = args.num_segments
    hyper_params['Learning Rate'] = args.lr
    hyper_params['Batch Size'] = args.batch_size
    hyper_params['Epochs'] = args.epochs
    
    solver = Solver(args.cpu, args.epochs, args.lr, args.batch_size,
                    args.resize, args.crop_size, args.num_segments, args.videos, args.weights_dir)
    solver.load_state()
    solver.load_labeldata()
    
    if not args.noresume:
        solver = solver.load_resume()
    
    if args.predict:
        solver.load_labeldata()
        solver.predict(args.predict)
        return
    
    for key in hyper_params.keys():
        print(f'{key}: {hyper_params[key]}')
    #experiment.log_parameters(hyper_params)
    
    if args.validate:
        solver.validate()
        return
    
    solver.save_labeldata()
    solver.train(not args.noresume)
    #experiment.end()


# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=str, default='')
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--resize', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--num_segments', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--noresume', action='store_true')
    parser.add_argument('--predict', type=str, default='')
    parser.add_argument('--validate', action='store_true')

    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.weights_dir):
        os.mkdir(args.weights_dir)
    
    main(args)

