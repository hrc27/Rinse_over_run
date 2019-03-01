#!/usr/bin/env python
# coding: utf-8

# # Creating Data

# In[36]:


import numpy as np


# In[61]:


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i+n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# In[62]:


raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]


# In[63]:


n_steps = 3


# In[64]:


X,y=split_sequence(raw_seq, n_steps)


# In[65]:


X


# In[66]:


y


# # Creating Model

# In[37]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# In[67]:


n_features = 1


# In[70]:


# one feature per timestep, three timesteps per sample, six samples total
X = X.reshape((X.shape[0], X.shape[1], n_features))


# In[72]:


X.shape[1]


# In[71]:


model = Sequential()
# input shape is ( number of timesteps per sample X number of features per sample)
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[84]:


model.fit(X, y, epochs=200, verbose=2)


# In[85]:


x_input = np.array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)


# In[86]:


yhat


# ## Stacked LSTM

# In[87]:


model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[88]:


model.fit(X, y, epochs=200, verbose=1)


# In[89]:


x_input = np.array([70, 80, 90, 100, 110, 120])
x_input = x_input.reshape((2, n_steps, n_features))


# In[90]:


yhat = model.predict(x_input, verbose=1)


# In[91]:


yhat


# ## Bidirectional LSTM

# In[3]:


from keras.layers import Bidirectional


# In[94]:


model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[96]:


model.fit(X, y, epochs=200, verbose=2)


# In[97]:


yhat = model.predict(x_input, verbose=1)


# In[98]:


yhat


# # Multivariable Data Prep

# In[141]:


in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])


# In[142]:


# convert to [rows, columns] format
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))


# In[143]:


dataset = np.hstack((in_seq1, in_seq2, out_seq))


# In[144]:


dataset


# In[145]:


# split a multivariate sequence into samples
def split_sequences_multi(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_ix = i+n_steps
        if end_ix > len(sequences):
            break
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# In[146]:


n_steps = 3


# In[147]:


X, y = split_sequences_multi(dataset, n_steps)


# In[149]:


X.shape


# In[150]:


y


# ## Model

# In[151]:


model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(None, 2)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[152]:


x_input = np.array([[80, 85], [90, 95], [100, 105]])
x_input = x_input.reshape((1, 3, 2))


# In[153]:


model.fit(X, y, epochs=200, verbose=1)


# In[154]:


yhat = model.predict(x_input, verbose=1)


# In[155]:


yhat


# # Rinse_over_Run Data

# In[2]:


import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import numpy as np
import feather


# In[3]:


train_limited = pd.read_feather('tmp/ROR_train_limited')


# In[4]:


train_labels = pd.read_feather('tmp/ROR_train_labels')


# In[5]:


train_limited.timestamp = train_limited.timestamp.view(int)


# In[6]:


ts_cols = [
    'timestamp',
    'process_id',
    'pipeline',
    'phase',
    'object_id',
    'supply_flow',
    'supply_pressure',
    'return_temperature',
    'return_conductivity',
    'return_turbidity',
    'return_flow',
    'tank_level_pre_rinse',
    'tank_level_caustic',
    'tank_level_acid',
    'tank_level_clean_water',
    'tank_temperature_pre_rinse',
    'tank_temperature_caustic',
    'tank_temperature_acid',
]


# In[7]:


train_converted = pd.get_dummies(train_limited[ts_cols])


# In[9]:


idx_array = train_converted.process_id.unique()


# In[10]:


p01 = train_converted[train_converted.process_id == idx[0]]


# In[17]:


p02 = train_converted[train_converted.process_id == idx[1]]


# In[18]:


p03 = train_converted[train_converted.process_id == idx[2]]


# In[19]:


a01 = p01.values


# In[20]:


a02 = p02.values


# In[21]:


a03 = p03.values


# In[22]:


a03.shape


# In[23]:


a02_buffer = np.zeros((a03.shape[0]-a02.shape[0], a01.shape[1]))


# In[24]:


a01_buff = np.zeros((a03.shape[0]-a01.shape[0], a01.shape[1]))


# In[25]:


a02 = np.vstack((a02, a02_buffer))


# In[26]:


a01 = np.vstack((a01, a01_buff))


# In[27]:


arr_X = np.vstack((a01, a02, a03))


# In[28]:


X = arr_X.reshape(3, a03.shape[0], a01.shape[1])


# In[29]:


X.shape


# In[34]:


n_features = 31


# In[79]:


def update_X_y(X, labels, idx):
    X_copy = X
    p_idx = train_converted[train_converted.process_id == idx_array[idx]]
    a_idx = p_idx.values
    print("Shape of a_idx is {}".format(a_idx.shape))
    print("Shape of X is {}".format(X.shape))
    if a_idx.shape[0] <= X.shape[1]:
        a_idx_buff = np.zeros((X.shape[1]-a_idx.shape[0], n_features))
        a_idx = np.vstack((a_idx, a_idx_buff))
        print("New a_idx shape is {}".format(a_idx.shape))
        X_melt = X.reshape((X.shape[0]*X.shape[1], n_features))
        print("X_melt shape is {}".format(X_melt.shape))
        arr_X = np.vstack((X_melt, a_idx))
        print("arr_X shape is {}".format(arr_X.shape))
        X = arr_X.reshape(idx + 1, X.shape[1], n_features)
        print("X shape is {}".format(X.shape))
    elif a_idx.shape[0] > X.shape[1]:
        
        print("The new process is longer than the present matrix - Update your function, Holden")
    y_sub = labels[:idx+1]
    print("y shape is {}".format(y.shape))
    return X, y_sub


# In[83]:


X, y = update_X_y(X, train_labels, 6)


# In[80]:


y = train_labels[:6]


# ## Model

# In[81]:


model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(a03.shape[0], a01.shape[1])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[82]:


model.fit(X, y, epochs=10, verbose=1)


# In[ ]:




