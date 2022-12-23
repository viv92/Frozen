'''
--- Program implementing the Frozen method with Resnet as the image encoder

-- Features:
1. Resnet as the image encoder is the only trainable component, while the T5 encoder and decoder are frozen during training
2. image encoder provides the image embedding which are feeded to the T5 encoder (by-passing the T5 encoder's embedding layer)
3. Any text inputs to the T5 encoder (not part of the input prompt during training on the image captioning task, but may be part of the prompt during inference on any other task) do go through the T5 encoder's embedding layer to convert text tokens (obtained from T5 tokenizer) to embeddings.
4. The T5 encoder's output is used to condition the T5 decoder via cross-attention. The T5 decoder predicts the output / target text in a teacher-forcing regime.
5. Loss is cross-entropy loss over the predicted target text. The gradients are backpropped through the T5 decoder, T5 decoder and back to image encoder. Only the image encoder params are updated.

-- Todos / Questions:
1. What are the init weights of the Resnet

'''

import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import matplotlib.pyplot as plt 
import cv2 
import math 
from copy import deepcopy
from tqdm import tqdm 
import json 

# import T5 
from transformers import T5Tokenizer, T5ForConditionalGeneration


# forward hook for reading resnet penultimate layer logits
def forward_hook(module, input, output):
    global resnet_avgpool_output
    resnet_avgpool_output = output

# class for image embeddings - obtained from resnet
class ImageEmbeddings(nn.Module):
    def __init__(self, hook_fn, d_model):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        self.resnet.avgpool.register_forward_hook(hook_fn)
        self.proj = nn.Linear(2048, d_model * 2, bias=False) # d_model * 2 because each image is supposed to constitute embeddings of seq_len = 2 (according to the paper)
    def forward(self, imgs): # imgs.shape: [b,c,w,h]
        batch_size = imgs.shape[0]
        _ = self.resnet(imgs)
        emb = resnet_avgpool_output # emb.shape: [b, 2048, 1, 1]
        emb = emb.flatten(start_dim=1, end_dim=-1) # emb.shape: [b, 2048]
        emb = self.proj(emb) # emb.shape: [b, d_model * 2]
        emb = emb.reshape(batch_size, d_model, 2)
        emb = emb.permute(0, 2, 1) # emb.shape: [batch_size, 2, d_model]
        return emb


# utility function to load img and captions data 
# no need to calculate max_cap_len as T5 Tokenizer handles that
# no need to strip accents or lowercase caption as T5 tokenizer handles that
def load_data():
    imgs_folder = 'dataset_coco_val2017/images/'
    captions_file_path = 'dataset_coco_val2017/annotations/captions_val2017.json'
    captions_file = open(captions_file_path)
    captions = json.load(captions_file)
    img_dict, img_cap_dict = {}, {}
    for img in captions['images']:
        id, file_name = img['id'], img['file_name']
        img_dict[id] = file_name
    for cap in captions['annotations']:
        id, caption = cap['image_id'], cap['caption']
        # use img_name as key for img_cap_dict
        img = img_dict[id]
        img_cap_dict[img] = caption
    return img_cap_dict


# utility function to process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions - then obtain embeddings for them
def process_batch(minibatch, tokenizer, img_size, device):
    augmented_imgs = []
    img_files, captions = list(map(list, zip(*minibatch)))
    # tokenize captions 
    caption_tokens_dict = tokenizer(captions, return_tensors='pt', padding=True, truncation=True)
    # get augmented imgs
    imgs_folder = 'dataset_coco_val2017/images/'
    for img_filename in img_files:
        img_path = imgs_folder + img_filename
        img = cv2.imread(img_path, 1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.permute(2, 1, 0) # [w,h,c] -> [c,h,w]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize( int(1.25*img_size) ),  # image_size + 1/4 * image_size
            torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # imagenet stats for normalization
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # zero mean, unit std
        ])
        img = transforms(img)
        augmented_imgs.append(img)
    augmented_imgs = torch.stack(augmented_imgs, dim=0).to(device)
    caption_tokens_dict = caption_tokens_dict.to(device)
    return augmented_imgs, caption_tokens_dict


# function to forward prop through the image encoder and T5 model to calculate loss
def calculate_loss(image_encoder, tokenizer, t5_model, imgs, cap_tokens_dict, device):
    batch_size = imgs.shape[0]
    # obtain img embeddings
    img_embs = image_encoder(imgs) # img_embs.shape: [batch_size, 2, d_model]
    # feed img_embs to t5 encoder to get encoder output
    enc_out = t5_model.encoder(inputs_embeds=img_embs).last_hidden_state # enc_out.shape: [batch_size, 2, d_model]
    # extract cap tokens and attn_mask from cap_tokens_dict
    cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
    # shift cap_tokens right (pre-pend start token) - as input to decoder is expected to be right shifted and starting with pad token (used as start token by T5)
    start_token_id = tokenizer(tokenizer.pad_token, return_tensors='pt', padding=False, truncation=True).input_ids
    start_token_id = start_token_id[:, 0] # trim end token appended by the tokenizer
    start_token_id = start_token_id.expand(batch_size, -1).to(device) # start_token_id.shape: [batch_size, 1]
    cap_tokens_rshifted = torch.cat((start_token_id, cap_tokens), dim=-1) # cap_tokens_rshifted.shape: [batch_size, seq_len+1]
    cap_tokens_rshifted = cap_tokens_rshifted[:, :-1] # cap_tokens_rshifted.shape: [batch_size, seq_len]
    # feed cap tokens to t5 decoder to get decoder output
    dec_out = t5_model.decoder(input_ids=cap_tokens_rshifted, attention_mask=cap_attn_mask, encoder_hidden_states=enc_out).last_hidden_state # dec_out.shape: [batch_size, seq_len, d_model]
    # get scores from dec_out
    scores = t5_model.lm_head(dec_out) # scores.shape: [batch_size, seq_len, vocab_size]
    scores = scores.permute(0, 2, 1) # scores.shape: [batch_size, vocab_size, seq_len] - required for crossEntropyLoss
    # create targets = cap_tokens (unshifted)
    targets = cap_tokens # targets.shape: [batch_size, seq_len]
    # cross entropy loss 
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(scores, targets)
    # calculate batch accuracy 
    pred_cap_tokens = torch.argmax(scores, dim=1) # shape: [batch_size, seq_len]
    batch_accuracy = (pred_cap_tokens == cap_tokens).float().mean() * 100
    return loss, batch_accuracy


# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() # clip is only used for inference
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer

# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on whatever device the model was training on
def save_ckpt(checkpoint_path, model, optimizer, scheduler=None):
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)


### main ###
if __name__ == '__main__':

    # hyperparams
    img_size = 224 # resize for resnet
    d_model = 768 # d_model for T5 (required for resnet proj head)
    max_seq_len = 512 # required to init T5 Tokenizer
    batch_size = 16
    lr = 3e-4
    num_epochs = 1000 * 20 
    random_seed = 1010

    t5_model_name = 't5-base'

    checkpoint_path = 'ckpts_frozen_resnet/latest.pt' # path to a save and load checkpoint of the trained resnet
    resume_training_from_ckpt = True 

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data and create img_cap_dict
    img_cap_dict = load_data()

    # create dataset from img_cap_dict
    dataset = []
    for i, (k, v) in enumerate(img_cap_dict.items()):
        dataset.append([k, v])
    dataset_len = len(dataset)
    # free memory occupied by img_cap_dict as its no longer needed
    del img_cap_dict

    # init image encoder model (resnet)
    image_encoder = ImageEmbeddings(forward_hook, d_model).to(device)

    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=max_seq_len)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

    # init optimizer 
    optimizer = torch.optim.Adam(params=image_encoder.parameters(), lr=lr, betas=(0.9, 0.95))

    # load checkpoint to resume training from
    if resume_training_from_ckpt:
        image_encoder, optimizer = load_ckpt(checkpoint_path, image_encoder, optimizer=optimizer, device=device, mode='train')

    # train loop
    for ep in tqdm(range(num_epochs)):

        # fetch minibatch
        idx = np.arange(dataset_len)
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        minibatch = [dataset[i] for i in idx]

        # process minibatch - convert img_filenames to augmented_imgs and convert caption_text to tokenized_captions
        # note that we don't create embeddings yet, since that's done by the image_encoder and T5 model
        imgs, cap_tokens_dict = process_batch(minibatch, t5_tokenizer, img_size, device) # imgs.shape:[batch_size, 3, 64, 64], captions.shape:[batch_size, max_seq_len]

        # calculate loss
        loss, batch_accuracy = calculate_loss(image_encoder, t5_tokenizer, t5_model, imgs, cap_tokens_dict, device)

        # update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep+1) % (num_epochs // 20) == 0:
            # print metrics
            print('ep:{} \t loss:{:.3f} \t batch_accuracy:{:.3f}'.format(ep, loss.item(), batch_accuracy.item()))

            # save model checkpoint 
            save_ckpt(checkpoint_path, image_encoder, optimizer)