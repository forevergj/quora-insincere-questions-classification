import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self,embedding_matrix,max_features,embed_size,maxlen,num_filters,filter_size_list):
        super(Network, self).__init__()
        self.embedding_matrix = embedding_matrix
        self.max_features = max_features#95000
        self.embed_size = embed_size#300
        self.maxlen = maxlen#70
        self.num_filters = num_filters
        self.filter_sizes = filter_size_list#filter_sizes = [1,2,3,5]
        self.dropoutrate = 0.1

        self.embedding = nn.Embedding(self.max_features, self.embed_size,
                                      )
        self.embedding.weight.data.copy_(torch.from_numpy(self.embedding_matrix))#[?,1,70,300]
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=num_filters,
                                             kernel_size=(self.filter_sizes[0], embed_size)),#[?,num_filters(36),70,1]
                                   nn.ELU(),
                                   nn.MaxPool2d(kernel_size=(maxlen - self.filter_sizes[0] + 1, 1)))#kernelsize[70,1],shape[?,36,1,1]
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_filters,
                                             kernel_size=(self.filter_sizes[1], embed_size)),#[?,num_filters(36),69,1]
                                   nn.Dropout(self.dropoutrate),
                                   nn.ELU(),
                                   nn.MaxPool2d(kernel_size=(maxlen - self.filter_sizes[1] + 1, 1)))#[?,num_filters(36),1,1]
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_filters,
                                             kernel_size=(self.filter_sizes[2], embed_size)),
                                   nn.Dropout(self.dropoutrate),
                                   nn.ELU(),
                                   nn.MaxPool2d(kernel_size=(maxlen - self.filter_sizes[2] + 1, 1)))#[?,num_filters(36),1,1]
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=num_filters,
                                             kernel_size=(self.filter_sizes[3], embed_size)),
                                   nn.Dropout(self.dropoutrate),
                                   nn.ELU(),
                                   nn.MaxPool2d(kernel_size=(maxlen - self.filter_sizes[3] + 1, 1)))#[?,num_filters(36),1,1]
        self.out = nn.Linear(36*4, 2)

    def forward(self, x):
        em = self.embedding(x)#[512,70,300]
        em = torch.from_numpy(np.expand_dims(em.detach().numpy(),axis=1))#[512,1,70,300]
        x1 = self.conv1(em)#[?,36,1,1]
        x2 = self.conv2(em)
        x3 = self.conv3(em)
        x4 = self.conv4(em)
        self.max_pool = torch.cat((x1,x2,x3,x4),2)#axis = 2连接maxlen维的数据
        self.flatten = self.max_pool.view(self.max_pool.shape[0],-1)
        output = F.sigmoid(self.out(self.flatten))
        return output
