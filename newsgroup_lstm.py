
# coding: utf-8

# In[72]:

from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.layers import Embedding, Dense, LSTM, GRU
from keras.models import Sequential


# In[33]:

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# In[2]:

categories = ['comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast']


# In[55]:

data = fetch_20newsgroups(shuffle=True, subset='train', categories=categories)


# In[54]:

test = fetch_20newsgroups(shuffle=True, subset='test', categories=categories)


# In[27]:

news, news_topics, _classes = data.data, data.target, data.target_names
encoded_labels = [_classes.index(topic) for topic in news_topics]


# In[31]:




# In[35]:

# import string
# def translate_non_alphanumerics(to_translate, translate_to='_'):
#     not_letters_or_digits = string.punctuation #u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
#     translate_table = string.maketrans(not_letters_or_digits,
#                                        translate_to
#                                          *len(not_letters_or_digits))
#     translate_table = translate_table.decode("latin-1")
#     return to_translate.translate(translate_table)

# for i, item in enumerate(news):
#     news[i] = translate_non_alphanumerics(item)


# In[76]:

nb_words = 10000
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(news)
sequences = Tokenizer.texts_to_sequences(tokenizer, news)


# In[81]:

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[32]:

max_seq_len = 1000
input_data = pad_sequences(sequences, maxlen=max_seq_len)
one_hot_labels = keras.utils.to_categorical(news_topics)

print('Shape of data tensor:', input_data.shape)
print('Shape of label tensor:', one_hot_labels.shape)


# In[107]:

Xtrain, Xvalid, ytrain, yvalid = train_test_split(input_data, one_hot_labels, test_size=0.2)


# In[94]:

len(word_index)


# In[113]:

embedding_vector_length = 64
model = Sequential()
model.add(Embedding(len(word_index), embedding_vector_length, input_length=max_seq_len, init='glorot_normal', 
                    W_regularizer=keras.regularizers.l2(0.01)))
model.add(LSTM(100, dropout_W=0.25))
model.add(Dense(7, activation='softmax'))


# In[114]:

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[108]:

Xtrain.shape, ytrain.shape, Xvalid.shape, yvalid.shape


# In[ ]:

model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), nb_epoch=3, batch_size=128)


# In[ ]:




# In[ ]:



