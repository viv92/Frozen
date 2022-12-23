### Program implementing Vision Transformer (ViT)

## Features:
# 1. Image of shape [H, W, C] is split into a sequence of patches of shape [N, P, P, C]; where seq_len = N = H*W/P*P (note that H % P and W % P should be zero)
# 2. Sequence of N patches is flattened into sequence of N tokens: [N, P, P, C] -> [N, P*P*C] (dim of each token = patch_dim = P*P*C - unlike in NLP where token_dim = 1)
# 3. Sequence of tokens is converted to patch embedding using nn.Linear(patch_dim, embed_dim). [We don't use nn.Embedding(vocab_size, embed_dim) as there is no meaning of vocab_size for image patches]
# 4. A dummy CLS token is appended to sequence of patch embeddings (as placeholder for final representation at transformer_encoder output; in similar spirit to BERT)
# 5. Learnable positional embeddings are added to obtain final image embeddings (shape: [batch_size, N+1, embed_dim]), which are inputted to transformer encoder
# 6. encoder_out.shape: [batch_size, N+1, embed_dim]. Final image representation: encoded_img = encoder_out[:, 0] (output corresponding to CLS token).
# 7. encoded_img is then feeded to MLP for classification (or any other downstream task)
# 8. ViT trained on a lower image resolution can be used for images with higher resolution by increasing the seq_len (keeping the same patch_dim). Only modification required is the interpolation of learned positional embeddings (refer ViT paper for details)
# 9. No padding or causal masks needed.

## Todos / Questions:
# 1. handle imges of bigger resolution than what the model was trained for (this will make input_seq_len > self.seq_len in ImageEmbeddings)

import math
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from time import time

# import transformer modules 
from utils_transformer import *

# class for image embeddings
class ImageEmbeddings(nn.Module):
    def __init__(self, patch_dim, seq_len, d_model, dropout, device):
        super().__init__()
        self.patch_emb = nn.Linear(patch_dim, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(seq_len+1, d_model)) # seq_len + 1 for cls token
        self.cls_emb = nn.Parameter(torch.randn(1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.device = device
    def forward(self, x): # s.shape: [batch_size, seq_len, patch_dim]
        batch_size, input_seq_len = x.shape[0], x.shape[1]
        patch_emb = self.patch_emb(x) # patch_emb.shape: [batch_size, seq_len, d_model]
        # broadcast cls embeddings and concate to patch_emb
        cls_emb = self.cls_emb.unsqueeze(0) # cls_emb.shape: [1, 1, d_model]
        cls_emb = cls_emb.expand(batch_size, -1, -1) # cls_emb.shape: [batch_size, 1, d_model]
        patch_emb = torch.cat((cls_emb, patch_emb), dim=1) # patch_emb.shape: [batch_size, seq_len+1, d_model]
        # add positional embeddings
        pos_emb = self.pos_emb
        pos_emb = pos_emb.unsqueeze(0) # pos_emb.shape: [1, seq_len+1, d_model]
        pos_emb = pos_emb.expand(batch_size, -1, -1) # pos_emb.shape: [batch_size, seq_len+1, d_model]
        final_emb = self.dropout( self.norm(patch_emb + pos_emb) )
        # final_emb = final_emb * math.sqrt(self.d_model) - think this is not needed since we are taking the norm
        return final_emb


# class implemeting the vision transformer
class ViT(nn.Module):
    def __init__(self, patch_size, img_emb, encoder, d_model, out_dim):
        super().__init__()
        self.patch_size = patch_size
        self.img_emb = img_emb
        self.encoder = encoder
        self.mlp = nn.Linear(d_model, out_dim)

    def img_to_patch_seq(self, img):
        b,c,h,w = img.shape
        p = self.patch_size
        assert (h % p == 0) and (w % p == 0)
        seq_len = (h * w) // (p * p)
        n_rows, n_cols = h//p, w//p
        patch_seq = []
        for row in range(n_rows):
            for col in range(n_cols):
                i = row * p
                j = col * p
                patch = img[:, :, i:i+p, j:j+p]
                patch = torch.flatten(patch, start_dim=1, end_dim=-1) # patch.shape: [b, patch_dim]
                patch_seq.append(patch)
        patch_seq = torch.stack(patch_seq, dim=0) # patch_seq.shape: [seq_len, b, patch_dim]
        patch_seq = patch_seq.transpose(0, 1) # patch_seq.shape: [b, seq_len, patch_dim]
        # assert seq_len == patch_seq.shape[1]
        # assert patch_dim == patch_seq.shape[-1]
        return patch_seq

    def encode(self, img):
        # convert image into flat patch_seq
        patch_seq = self.img_to_patch_seq(img) # patch_seq.shape: [batch_size, seq_len, patch_dim]
        # forward prop through transformer encoder
        encoder_out = self.encoder(self.img_emb(patch_seq))
        return encoder_out

    def forward(self, img): # img.shape: [batch_size, C, H, W,]
        encoder_out = self.encode(img) # encoder_out.shape: [batch_size, seq_len, d_model]
        vit_out = self.mlp(encoder_out[:, 0, :]) # vit_out.shape: [batch_size, out_dim]
        return vit_out, encoder_out


# caller function to instantiate the ViT model, using the defined hyperparams as input
def init_vit(patch_size, patch_dim, seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, out_dim, device):
    attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout) # multi head attention block
    ff = FeedForward(d_model, d_ff, dropout) # feed forward block for each encoder / decoder block
    img_emb = ImageEmbeddings(patch_dim, seq_len, d_model, dropout, device) # image embeddings block to obtain sequence of embeddings from sequence of patch tokens
    encoder_layer = EncoderLayer(deepcopy(attn), deepcopy(ff), d_model, dropout) # single encoder layer
    encoder = Encoder(encoder_layer, n_layers, d_model) # encoder = stacked encoder layers
    model = ViT(patch_size, img_emb, encoder, d_model, out_dim) # the ViT model
    # initialize params - Xavier initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# function to get test loss and accuracy
def get_test_loss_accuracy(vit, testloader, criterion):
    correct = 0
    total = 0
    with torch.no_grad():
        epoch_losses = []
        running_loss = 0
        for i, data in enumerate(testloader):
            imgs, labels = data[0].to(device),data[1].to(device)
            vit_out, enc_out = vit(imgs)
            loss = criterion(vit_out, labels)
            _, predicted = torch.max(vit_out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss
            if i % 2000 == 1999:
                epoch_losses.append(running_loss)
                running_loss = 0
        test_accuracy = correct/float(total)
        test_loss = sum(epoch_losses)/len(epoch_losses)
    return test_loss.item(), test_accuracy


# main - training ViT on CIFAR
if __name__ == '__main__':
    # hyperparams
    img_size = 256
    img_channels = 3
    patch_size = 16 # necessary that img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)
    seq_len = (img_size // patch_size) ** 2
    out_dim = 10 # num_classes for CIFAR
    d_model = 512
    d_k = 64
    d_v = 64
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    batch_size = 4
    lr = 1e-4
    num_epochs = 5
    random_seed = 10

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # init model
    vit = init_vit(patch_size, patch_dim, seq_len, d_model, d_k, d_v, n_heads, n_layers, d_ff, dropout, out_dim, device).to(device)

    # init optimizer
    optimizer = torch.optim.Adam(params=vit.parameters(), lr=lr)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # load CIFAR dataset
    resize_shape = (img_size, img_size)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
        torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
    ])
    trainset = torchvision.datasets.CIFAR10(root='./dataset_cifar', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./dataset_cifar', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # train
    results_train_loss = []
    results_test_loss = []
    results_test_accuracy = []
    start_time = time()
    for epoch in range(num_epochs):
        running_loss = 0
        epoch_loss = 0
        for i, data in tqdm(enumerate(trainloader, 0)):
            imgs, labels = data[0].to(device), data[1].to(device) # imgs.shape: [b, c, h, w]

            optimizer.zero_grad()
            vit_out, enc_out = vit(imgs)
            loss = loss_fn(vit_out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            if i % 2000 == 1999:
                test_loss, test_accuracy = get_test_loss_accuracy(vit, testloader, loss_fn)
                # print('epoch:{} \t i:{} \t train_loss:{} \t test_loss:{} \t test_accuracy:{}'.format(epoch, i, running_loss/2000, test_loss/2000, test_accuracy))
                results_train_loss.append(running_loss/2000)
                results_test_loss.append(test_loss/2000)
                results_test_accuracy.append(test_accuracy)
                running_loss = 0

        print('epoch_loss: ', epoch_loss / i)

    end_time = time()
    time_taken = end_time - start_time

    # plot results
    x = np.arange(len(results_test_loss))
    plt.plot(x, results_train_loss, label='train_loss')
    plt.plot(x, results_test_loss, label='test_loss')
    plt.plot(x, results_test_accuracy, label='test_accuracy')
    plt.legend()
    plt.title('time_taken: ' + str(time_taken))
    plt.savefig('vit_cifar_learning_curves.png')
