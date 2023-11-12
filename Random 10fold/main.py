from tensorflow.keras import Model, layers, initializers
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from keras.layers import Dense,Input,Flatten,Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, concatenate
from keras.preprocessing.text import Tokenizer,one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Model
import pandas as pd
import os
import numpy as np
from keras.layers import LSTM,LayerNormalization
from keras.layers import GRU
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.layers import Bidirectional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from keras_self_attention import SeqSelfAttention
# from keras.utils import to_categorical
from keras.layers import CuDNNLSTM,CuDNNGRU

from keras.layers import Input,Dense,Dropout,Conv2D,MaxPool2D,Flatten,GlobalAvgPool2D,concatenate,BatchNormalization,Activation,Add,ZeroPadding2D,Lambda
from keras.layers import Dense,Dropout,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Activation

from sklearn.model_selection import KFold
import random

# Model training using CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.config.list_physical_devices('GPU')

# auc_best = 0
# fpr_best = 0
# tpr_best = 0
best_acc = 0
x_train = None
y_train = None
x_test = None
y_test = None
# auc_best_list = {}
accuracy_best_list = {}

def machine_learning(pro_x_train, pro_y_train, pro_x_test, pro_y_test, number):
    # global auc_best, fpr_best, tpr_best
    global x_test, y_test, x_train, y_train
    global best_acc
    global accuracy_best_list
    Test_label = []
    best_acc = 0

    y_train =  pro_y_train
    pos_train_len = len(pro_x_train)
    y_test = pro_y_test


    # Merge datasets
    pro_x_data  = pd.concat([pro_x_train,pro_x_test],ignore_index= True )
    pro_y_data  = pd.concat([y_train,y_test],ignore_index= True )


    # Create a corresponding label using one-hot encoding for each RNA sequence
    for i in range(0, len(pro_y_data)):

            if(pro_y_data[i] == '5S_rRNA'):
                Test_label.append([1,0,0,0,0,0,0,0,0,0,0,0,0])
            elif(pro_y_data[i] == '5_8S_rRNA'):
                Test_label.append([0,1,0,0,0,0,0,0,0,0,0,0,0])
            elif(pro_y_data[i] == 'tRNA'):
                Test_label.append([0,0,1,0,0,0,0,0,0,0,0,0,0])
            elif(pro_y_data[i] == 'ribozyme'):
                Test_label.append([0,0,0,1,0,0,0,0,0,0,0,0,0])
            elif(pro_y_data[i] == 'CD-box'):
                Test_label.append([0,0,0,0,1,0,0,0,0,0,0,0,0])
            elif(pro_y_data[i] == 'miRNA'):
                Test_label.append([0,0,0,0,0,1,0,0,0,0,0,0,0])
            elif(pro_y_data[i] == 'Intron_gpI'):
                Test_label.append([0,0,0,0,0,0,1,0,0,0,0,0,0])
            elif(pro_y_data[i] == 'Intron_gpII'):
                Test_label.append([0,0,0,0,0,0,0,1,0,0,0,0,0])
            elif(pro_y_data[i] == 'HACA-box'):
                Test_label.append([0,0,0,0,0,0,0,0,1,0,0,0,0])
            elif(pro_y_data[i] == 'riboswitch'):
                Test_label.append([0,0,0,0,0,0,0,0,0,1,0,0,0])
            elif(pro_y_data[i] == 'IRES'):
                Test_label.append([0,0,0,0,0,0,0,0,0,0,1,0,0])
            elif(pro_y_data[i] == 'leader'):
                Test_label.append([0,0,0,0,0,0,0,0,0,0,0,1,0])
            elif(pro_y_data[i] == 'scaRNA'):
                Test_label.append([0,0,0,0,0,0,0,0,0,0,0,0,1])
            else:
                print(i)
                print(pro_y_data[i])


    # Represent each RNA sequence using a 2-mer(k-mer method) encoding
    K = 2
    str_array = []
    loopcnt = 0
    for i in pro_x_data:
        seq_str = str(i)
        seq_str = seq_str.strip('[]\'')
        t=0
        l=[]
        for index in range(len(seq_str)):
            t=seq_str[index:index+K]
            if (len(t))==K:
                l.append(t)
        str_array.append(l)



    # Convert RNA sequences into integer sequences using a tokenizer
    # Then, ensure that each RNA sequence has a uniform length of 224 by either truncating or padding with zeros
    tokenizer = Tokenizer(num_words = 30000)
    tokenizer.fit_on_texts(str_array)
    sequences = tokenizer.texts_to_sequences(str_array)
    sequences = pad_sequences(sequences,maxlen = 224,padding = "post")
    # sequences = pad_sequences(sequences,maxlen = 200)
    sequences = np.array(sequences)


    # str_array2 = []

    # for i in sequences:
    #     str_array1 = []
    #     for char in i:
    #         if char == "A":
    #             str_array1.append([1,0,0,0,1,1,1])
    #         elif char == "U":
    #             str_array1.append([0,0,0,1,0,1,0])
    #         elif char == "C":
    #             str_array1.append([0,1,0,0,0,0,1])
    #         elif char == "G":
    #             str_array1.append([0,0,1,0,1,0,0])
    #     str_array2.append(str_array1)

    # sequences = str_array2


    x_train,x_test = sequences[:pos_train_len],sequences[pos_train_len:]
    y_train,y_test = Test_label[:pos_train_len],Test_label[pos_train_len:]
    
    
    # # convert to one hot
    # sequences = to_categorical(sequences, num_classes=5)
    # print(sequences[-2])
    # print(type(sequences))
    # print(sequences.shape)

    # sequences = np.delete(sequences, 0, axis=2)
    # print(sequences[-2])
    # print(sequences.shape)


    ## Shuffle the data
    # t = time.time()
    # my_time = int(round(t * 1000)) % 2147483648
    # print(my_time)
    # np.random.seed(my_time)
    # # np.random.seed(10)
    
    # print(x_train[:10])
    # print(y_test[:10])

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    # x_shuffled = x_train[shuffle_indices]
    # y_shuffled = y_train[shuffle_indices]


    # Convert the data into word vectors using word embeddings
    sequence_input = Input(shape = (224
    ,))
    embedding_layer = Embedding(30000,
                                16,
                                input_length = 224)
    embedded_sequences = embedding_layer(sequence_input)
    

    # Networdk model
    embedded_sequences = BatchNormalization(epsilon=1e-6)(embedded_sequences) 

    # The stem section of this paper employs both LSTM and CNN modules,
    # allowing for the simultaneous extraction of sequence and structural features
    stem = Conv1D(filters = 96,kernel_size = 8,padding="same",activation = "gelu")(embedded_sequences)
    lstm = Bidirectional(CuDNNLSTM(16,return_sequences = True))(embedded_sequences)
    lstm = layers.Activation("gelu")(lstm)
    stem = concatenate([stem,lstm],axis = 2)
    stem = BatchNormalization(epsilon=1e-6)(stem)
    stem = Dropout(0.5)(stem)

    # downsample1 =  stem[:,0::2,:]
    # downsample2 =  stem[:,1::2,:]
    # stem  = concatenate([downsample2,downsample1],axis = 2)

    # Fully Connected Module: It maps and compresses the features, 
    # thereby filtering valuable information from the feature set
    MLP_1  = Dense(int(128 * 2.0), name="Dense_0",kernel_initializer= initializers.GlorotUniform(),
    bias_initializer = initializers.RandomNormal(stddev=1e-6))(stem)
    Activation_1 = layers.Activation("gelu")(MLP_1)
    Activation_1 = BatchNormalization(epsilon=1e-6)(Activation_1)
    Dropout_2 = Dropout(0.2)(Activation_1)
    MLP_2  = Dense(int(128* 1.0), name="Dense_1",kernel_initializer= initializers.GlorotUniform(),
    bias_initializer = initializers.RandomNormal(stddev=1e-6))(Dropout_2)
    MLP_2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(MLP_2,stem)
    stem = stem + MLP_2


    #Multi-Window Convolutional Structure: It can extract RNA structural features of different sizes
    cnn1_24 = Conv1D(filters = 32,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(stem) 
    cnn1_16 = Conv1D(filters = 64,kernel_size = 16,strides=1,activation = "gelu",padding='same')(stem)
    cnn1_10 = Conv1D(filters = 32,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(stem)
    merge_fla_1  = concatenate([cnn1_24,cnn1_16,cnn1_10],axis = 2)
    merge_fla_1 = BatchNormalization(epsilon=1e-6)(merge_fla_1)
    merge_fla_1 = Dropout(0.2)(merge_fla_1)
    
    
    cnn2_24 = Conv1D(filters = 32,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_1)
    cnn2_16 = Conv1D(filters = 64,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_1)
    cnn2_10 = Conv1D(filters = 32,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_1)
    merge_fla_2  = concatenate([cnn2_24,cnn2_16,cnn2_10],axis = 2)
    merge_fla_2 = merge_fla_2 + merge_fla_1
    merge_fla_2 = BatchNormalization(epsilon=1e-6)(merge_fla_2)
    merge_fla_2 = Dropout(0.2)(merge_fla_2)
    

    cnn3_24 = Conv1D(filters = 32,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_2)
    cnn3_16 = Conv1D(filters = 64,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_2)
    cnn3_10 = Conv1D(filters = 32,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_2)
    merge_fla_3  = concatenate([cnn3_24,cnn3_16,cnn3_10],axis = 2)
    merge_fla_3 = merge_fla_2 + merge_fla_3
    merge_fla_3 = BatchNormalization(epsilon=1e-6)(merge_fla_3)
    merge_fla_3 = Dropout(0.2)(merge_fla_3) 
    
    cnn4_24 = Conv1D(filters = 32,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_3)
    cnn4_16 = Conv1D(filters = 64,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_3)
    cnn4_10 = Conv1D(filters = 32,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_3)
    merge_fla_4  = concatenate([cnn4_24,cnn4_16,cnn4_10],axis = 2)
    # A residual network structure with multi-head attention mechanisms between network layers
    merge_fla_4 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(merge_fla_1,merge_fla_4)
    merge_fla_4 = merge_fla_1 + merge_fla_4
    merge_fla_4 = BatchNormalization(epsilon=1e-6)(merge_fla_4) 
    merge_fla_4 = Dropout(0.2)(merge_fla_4) 


    # Downsampling Method: This approach, in contrast to using pooling and similar methods, 
    # can retain a greater amount of the original information in the input features
    downsample1 =  merge_fla_4[:,0::2,:]
    downsample2 =  merge_fla_4[:,1::2,:]
    merge_fla_4  = concatenate([downsample2,downsample1],axis = 2)


    cnn5_24 = Conv1D(filters = 64,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_4)
    cnn5_16 = Conv1D(filters = 128,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_4)
    cnn5_10 = Conv1D(filters = 64,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_4)
    merge_fla_5  = concatenate([cnn5_24,cnn5_16,cnn5_10],axis = 2)
    merge_fla_5 = merge_fla_5 + merge_fla_4 
    merge_fla_5 = BatchNormalization(epsilon=1e-6)(merge_fla_5)
    merge_fla_5 = Dropout(0.5)(merge_fla_5) 


    # cnn6_24 = Conv1D(filters = 64,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_5)
    # cnn6_16 = Conv1D(filters = 128,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_5)
    # cnn6_10 = Conv1D(filters = 64,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_5)
    # merge_fla_6  = concatenate([cnn6_24,cnn6_16,cnn6_10],axis = 2)
    # merge_fla_6 = merge_fla_6 + merge_fla_4 
    # merge_fla_6 = Dropout(0.2)(merge_fla_6)
    
     
    cnn7_24 = Conv1D(filters = 64,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_5)
    cnn7_16 = Conv1D(filters = 128,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_5)
    cnn7_10 = Conv1D(filters = 64,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_5)
    merge_fla_7  = concatenate([cnn7_24,cnn7_16,cnn7_10],axis = 2)
    merge_fla_7 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(merge_fla_4,merge_fla_7)
    merge_fla_7 = merge_fla_7 + merge_fla_4
    merge_fla_7 = BatchNormalization(epsilon=1e-6)(merge_fla_7)
    merge_fla_7 = Dropout(0.2)(merge_fla_7) 


    cnn8_24 = Conv1D(filters = 64,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_7)
    cnn8_16 = Conv1D(filters = 128,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_7)
    cnn8_10 = Conv1D(filters = 64,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_7)
    merge_fla_8  = concatenate([cnn8_24,cnn8_16,cnn8_10],axis = 2)
    merge_fla_8 = merge_fla_7 + merge_fla_8 
    merge_fla_8 = BatchNormalization(epsilon=1e-6)(merge_fla_8)
    merge_fla_8 = Dropout(0.2)(merge_fla_8) 


    cnn9_24 = Conv1D(filters = 64,kernel_size = 18,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_8)
    cnn9_16 = Conv1D(filters = 128,kernel_size = 16,strides=1,activation = "gelu",padding='same')(merge_fla_8)
    cnn9_10 = Conv1D(filters = 64,kernel_size = 10,strides=1,activation = "gelu",padding='same',dilation_rate=2)(merge_fla_8)
    merge_fla_9  = concatenate([cnn9_24,cnn9_16,cnn9_10],axis = 2)
    merge_fla_9 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(merge_fla_7,merge_fla_9)
    merge_fla_9 = merge_fla_7 + merge_fla_9
    merge_fla_9 = BatchNormalization(epsilon=1e-6)(merge_fla_9)


    merge_fla  = Flatten()(merge_fla_9)
    merge_fla = Dropout(0.5)(merge_fla)


    merge = Dense(200,activation = "sigmoid")(merge_fla)
    merge = Dropout(0.5)(merge)
    preds = Dense(13,activation = "softmax")(merge)
    model = Model(sequence_input,preds)

    model.summary()


    # model.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
    # model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=[auc])
    model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=['accuracy'])

    # my_callbacks = [EarlyStopping(monitor='auc_roc', patience=300, verbose=1, mode='max')]

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs={}):
            global best_acc
            epoch_pred = self.model.predict(x_test)
            for i in range(len(epoch_pred)):
                    max_value=max(epoch_pred[i])
                    for j in range(len(epoch_pred[i])):
                        if max_value==epoch_pred[i][j]:
                            epoch_pred[i][j]=1
                        else:
                            epoch_pred[i][j]=0
            print()
            print("epoch[",epoch+1,"].pred:\n", classification_report(y_test, epoch_pred, digits=5))
            my_valacc = logs.get('val_accuracy')
            print()
            if my_valacc > best_acc:
                best_acc = my_valacc
            print("epoch[",epoch+1,"].val_accuracy:", my_valacc)
            print("epoch[",epoch+1,"].best_accuracy:", best_acc)
            print()
            print()

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(suppress=True)
    history = LossHistory()
    # history = AUCHistory()

    model.fit(x_train,y_train,
            batch_size = 200,
            # epochs = 100,
            epochs = 220,
            callbacks=[history],
            validation_data = (x_test,y_test))
    
    accuracy_best_list[number] = best_acc

    for i in accuracy_best_list.keys():
        print("best_acc[",i,"] = ", accuracy_best_list[i])
    avg = float(sum(accuracy_best_list.values())) / len(accuracy_best_list)
    print()
    print("best_acc[avg] = ", avg)
    print()
    print()


#Data loading
dataset_fname = './dataset.csv'
dataset = pd.read_csv(dataset_fname, header=None)

# Read RNA sequences and labels
features = dataset[1][:]
labels = dataset[2][:]



loop = 0

# Generate a permutation of the indices
indices = np.random.permutation(len(features))


shuffled_features = features[indices]
shuffled_labels = labels[indices]


# 10 fold cross validation
kf_outer = KFold(n_splits=10, shuffle=False)

for fold_idx, (train_indices, test_indices) in enumerate(kf_outer.split(shuffled_features)):
    train_features, test_features = shuffled_features.iloc[train_indices], shuffled_features.iloc[test_indices]
    train_labels, test_labels = shuffled_labels.iloc[train_indices], shuffled_labels.iloc[test_indices]

    # Invoke the machine_learning function for training and evaluation
    machine_learning(train_features, train_labels, test_features, test_labels, loop)
    loop += 1
