import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from net import Network
import torch.nn as nn
import torch.utils.data as Data
max_features = 95000
train_path = 'D:\dataset\kaggle\quora-insincere-questions-classification/train.csv'
test_path =  'D:\dataset\kaggle\quora-insincere-questions-classification/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_df, val_df = train_test_split(train_df, test_size=0.08, random_state=2018)
## fill up the missing values
train_X = train_df["question_text"].fillna("_##_").values
val_X = val_df["question_text"].fillna("_##_").values
test_X = test_df["question_text"].fillna("_##_").values

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
print('document ready')

## Tokenize the sentences
print('make word_list')
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))#统计train_x的word次数，并按大到小排列，列出索引，得到word2i，即tokenize.word_index
train_X = tokenizer.texts_to_sequences(train_X)#将text转换为索引序列
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

# maxlen = max(
#     [len(s) for s in train_X + val_X +test_X ])会死机
maxlen = 70
print('max_lenth of sentences:%s'%maxlen)
## Pad the sentences
print('pading......')
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)
print(train_X)
#shuffing
print('shuffing')
np.random.seed(2018)
trn_idx = np.random.permutation(len(train_X))
# val_idx = np.random.permutation(len(val_X))

train_X = train_X[trn_idx]
# val_X = val_X[val_idx]
train_y = train_y[trn_idx]
# val_y = val_y[val_idx]
print('loading glove')
EMBEDDING_FILE = 'D:\dataset\kaggle\quora-insincere-questions-classification\embeddings\glove.840B.300d\glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE,encoding='UTF-8'))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
filter_sizes = [1,2,3,5]
num_filters = 36


net = Network(embedding_matrix,max_features,embed_size,maxlen,num_filters,filter_sizes)
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
#train
print('training')
batch_size = 512

lenth = train_X.shape[0]
for step in range(int(lenth / batch_size)):
        dx = train_X[batch_size*step:batch_size* (step+1)]
        dy = np.array([train_y[batch_size*step:batch_size* (step+1)]])# gives batch data, normalize x when iterate train_loader
        dx = torch.LongTensor(dx)
        dy = torch.LongTensor(dy)
        dy = dy.squeeze(0)
        output = net(dx)                # cnn output
        loss = loss_func(output, dy)    # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step %20==0:
            val_idx = np.random.choice(val_X.shape[0], size=200, replace=False)
            val_X = val_X[val_idx]
            val_y = val_y[val_idx]
            test_output= net(torch.LongTensor(val_X))
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == val_y).astype(int).sum()) / float(val_y.shape[0])
            print('step: ', step, '| train loss: %.4f' % loss.data.numpy(), 'test accuracy: %.2f' % accuracy)
