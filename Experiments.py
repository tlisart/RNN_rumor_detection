"""
File description : Running experiments for varying parameters to find optimal
                   configuration
"""

import numpy as np
import os
import math as m
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN, GRU

###################################################################################
#                                                                                 #
#       RNN                                                                       #
#                                                                                 #
###################################################################################



#krange = list of k values (should only contain k values for which the data files exist)
#lrrange = list of learning rates to be tested
#dropoutrange = list of dropouts to be tested
#lambrange = list of lambdas for regularization to be tested
#arch = architecture type: 0 = SimpleRNN, 1 = LSTM, 2 = GRU
def run_experiment(krange, lrrange, dropoutrange, lambrange, arch): 
    #Load the preprocessed data
    with open('labels_train_onehot.npy', 'rb') as f:
        labels_train_onehot = np.load(f)

    with open('labels_test_onehot.npy', 'rb') as f:
        labels_test_onehot = np.load(f)

    with open('labels_val_onehot.npy', 'rb') as f:
        labels_val_onehot = np.load(f)

    # Fixed experiment parameters
    amount_runs = 5   # Runs per experiments

    maxEpochs = 100
    allEpochs = []
    opti = 'Adam'
    #dropout = 0.2      #Dropout goodpractice ~20%
    embeddin_size = 100

    # Has to be adapted to the way you handle K
    for k in krange :

        #loading data, files must exist (should implement exception handling)
        fileNameTrain = 'RNN_data_train' + str(k) + '.npy'
        fileNameTest = 'RNN_data_test' + str(k) + '.npy'
        fileNameVal = 'RNN_data_val' + str(k) + '.npy'


        if os.path.exists(fileNameTrain):
            with open(fileNameTrain, 'rb') as f:
                try:
                    RNN_data_train = np.load(f)
                except :
                    print("Error : RNN_data_train for k : " + str(k) + "doesn't exist")

        if os.path.exists(fileNameTest):
            with open(fileNameTest, 'rb') as f:
                try:
                    RNN_data_test = np.load(f)
                except :
                    print("Error : RNN_data_test for k : " + str(k) + "doesn't exist")

        if os.path.exists(fileNameVal):
            with open(fileNameVal, 'rb') as f:
                try:
                    RNN_data_val = np.load(f)
                except :
                    print("Error : RNN_data_valfor k : " + str(k) + "doesn't exist")

        #Reshaping data
        N = RNN_data_train.shape[1]
        k = RNN_data_train.shape[2]

        # Experiments through learning rates
        for learningRate in lrrange :
            #Experiments with different dropout values
            
            for dropout in dropoutrange :
                for lamb in lambrange :
                    count = 0

                    #Accuracies
                    test_accuracies = []
                    train_accuracies = []
                    val_accuracies = []
                    val_loss= []

                    # Architecture SimpleRNN -------------------------------------------------------
                    if arch == 0 :
                        while count < amount_runs :
                            # define the based sequential model
                            model = Sequential()
                            # RNN layers
                            model.add(Dense(embeddin_size, input_shape=(N,k),
                                            kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
                            model.add(SimpleRNN(N,
                                                input_shape = (N, embeddin_size),
                                                return_sequences=False,
                                                kernel_regularizer = tf.keras.regularizers.l2(lamb)))
                            model.add(Dropout(dropout)) #Dropout

                            # Output layer for classification
                            model.add(Dense(2, activation='softmax',
                                            kernel_regularizer = tf.keras.regularizers.l2(lamb)))
                            model.summary()

                            # tf.keras.callbacks.EarlyStopping(
                            #     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                            #     baseline=None, restore_best_weights=False
                            # )

                            model.compile(
                                loss='categorical_crossentropy',
                                #optimizer=tf.keras.optimizers.Adagrad(learning_rate=learningRate, initial_accumulator_value=0.1, epsilon=1e-07),
                                optimizer= tf.keras.optimizers.Adam(lr=learningRate, decay=1e-5),
                                #optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
                                #regularizer=tf.keras.regularizers.l2(l=lamb),
                                metrics=['accuracy'],
                            )

                            # Train and test the model
                            model_history = model.fit(RNN_data_train,
                                    labels_train_onehot,
                                    epochs=maxEpochs,
                                    batch_size=64,
                                    validation_data=(RNN_data_test, labels_test_onehot))

                            # Evaluate the model
                            pred = model.predict(RNN_data_val)
                            y_pred = np.argmax(pred, axis=1)
                            lab= np.argmax(labels_val_onehot, axis=1)

                            #Recording accuracies
                            test_accuracies.append(np.mean(y_pred ==lab))
                            train_accuracies.append(np.mean(model_history.history["accuracy"]))
                            val_accuracies.append(np.mean(model_history.history["val_accuracy"]))
                            val_loss.append(np.mean(model_history.history["loss"]))


                            print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))
                            count += 1

                    # Architecture LSTM -------------------------------------------------------
                    if arch == 1 :
                        while count < amount_runs :
                            # define the based sequential model
                            model = Sequential()
                            # RNN layers
                            model.add(Dense(embeddin_size, input_shape=(N,k),
                                            kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
                            model.add(LSTM(N,
                                        input_shape = (N, embeddin_size),
                                        return_sequences=False,
                                        kernel_regularizer = tf.keras.regularizers.l2(lamb)))
                            model.add(Dropout(dropout)) #Dropout

                            # Output layer for classification
                            model.add(Dense(2, activation='softmax',
                                            kernel_regularizer = tf.keras.regularizers.l2(lamb)))
                            model.summary()


                            # tf.keras.callbacks.EarlyStopping(
                            #     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                            #     baseline=None, restore_best_weights=False
                            # )

                            model.compile(
                                loss='categorical_crossentropy',
                                # optimizer=tf.keras.optimizers.Adagrad(learning_rate=learningRate, initial_accumulator_value=0.1, epsilon=1e-07),
                                optimizer= tf.keras.optimizers.Adam(lr=learningRate, decay=1e-5),
                                #optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
                                # regularizer=tf.keras.regularizers.l2(l=lamb),
                                metrics=['accuracy'],
                            )

                            # Train and test the model
                            model_history = model.fit(RNN_data_train,
                                    labels_train_onehot,
                                    epochs=maxEpochs,
                                    batch_size=64,
                                    validation_data=(RNN_data_test, labels_test_onehot))

                            # Evaluate the model
                            pred = model.predict(RNN_data_val)
                            y_pred = np.argmax(pred, axis=1)
                            lab= np.argmax(labels_val_onehot, axis=1)

                            #Recording accuracies
                            test_accuracies.append(np.mean(y_pred ==lab))
                            train_accuracies.append(np.mean(model_history.history["accuracy"]))
                            val_accuracies.append(np.mean(model_history.history["val_accuracy"]))
                            val_loss.append(np.mean(model_history.history["loss"]))


                            print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))
                            count += 1

                    # Architecure GRU --------------------------------------------------------------


                    if arch == 2 :
                        while count < amount_runs :
                            # define the based sequential model
                            model = Sequential()

                            model.add(Dense(embeddin_size, input_shape=(N,k), kernel_regularizer = tf.keras.regularizers.l2(lamb))) #Embedding layer
                            #model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=True, kernel_regularizer = tf.keras.regularizers.l2(lamb)))
                            #model.add(Dropout(dropout)) #Dropout
                            model.add(GRU(N,input_shape = (N, embeddin_size),return_sequences=False,kernel_regularizer = tf.keras.regularizers.l2(lamb)))
                            #model.add(Dropout(dropout)) #Dropout
                            # Output layer for classification
                            model.add(Dense(2, activation='softmax',kernel_regularizer = tf.keras.regularizers.l2(lamb)))
                            model.summary()
                            # tf.keras.callbacks.EarlyStopping(
                            #     monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                            #     baseline=None, restore_best_weights=False
                            # )

                            model.compile(
                                loss='categorical_crossentropy',
                                #optimizer=tf.keras.optimizers.Adagrad(learning_rate=learningRate, initial_accumulator_value=0.1, epsilon=1e-07),
                                optimizer= tf.keras.optimizers.Adam(lr=learningRate, decay=1e-5),
                                #optimizer=tf.keras.optimizers.RMSprop(lr=1e-3),
                                # regularizer=tf.keras.regularizers.l2(l=lamb),
                                metrics=['accuracy'],
                            )

                            # Train and test the model
                            model_history = model.fit(RNN_data_train,
                                    labels_train_onehot,
                                    epochs=maxEpochs,
                                    batch_size=64,
                                    validation_data=(RNN_data_test, labels_test_onehot))

                            # Evaluate the model
                            pred = model.predict(RNN_data_val)
                            y_pred = np.argmax(pred, axis=1)
                            lab= np.argmax(labels_val_onehot, axis=1)

                            #Recording accuracies
                            test_accuracies.append(np.mean(y_pred ==lab))
                            train_accuracies.append(np.mean(model_history.history["accuracy"]))
                            val_accuracies.append(np.mean(model_history.history["val_accuracy"]))
                            val_loss.append(np.mean(model_history.history["loss"]))

                            print("Accuracy={:.2f}".format(np.mean(y_pred ==lab)))
                            count += 1

                    E_test = 0
                    E_train = 0
                    E_val = 0
                    E_loss = 0

                    S_test = 0
                    S_train = 0

                    #Mean over testing accuracy
                    for i in range(len(test_accuracies)) :
                        E_test += test_accuracies[i]
                    E_test = E_test/len(test_accuracies)

                    #Mean over testing accuracy
                    for i in range(len(train_accuracies)) :
                        E_train += train_accuracies[i]
                    E_train = E_train/len(train_accuracies)

                    #Mean over val accuracy
                    for i in range(len(val_accuracies)) :
                        E_val += val_accuracies[i]
                    E_val = E_val/len(val_accuracies)

                    #Mean over loss value
                    for i in range(len(val_loss)) :
                        E_loss += val_loss[i]
                    E_loss = E_loss/len(val_loss)

                    #\sigma^2 over experiments - test
                    for i in range(len(test_accuracies)) :
                        S_test += (test_accuracies[i] - E_test)**2
                    S_test = S_test/(len(test_accuracies))

                    #\sigma^2 over experiments - train
                    for i in range(len(train_accuracies)) :
                        S_train += (train_accuracies[i] - E_train)**2
                    S_train = S_train/(len(train_accuracies))


                    print("E(Accuracy)" + str(E_test) + "Amount experiments : " + str(amount_runs))

                    # Saving results of experiment for Panda

                    """
                    opt arch MaxEpochs E(test_accuracy) S(test_accuracy) E(train_accuracy) S(train_accuracy) E(val_accuracy) E(loss) nb(experiments) lr embedding K reg dropout
                    """
                    archi =''

                    if(arch == 0) :
                        archi = 'SimpleRNN'
                    if(arch == 1) :
                        archi = 'LSTM'
                    if(arch == 2) :
                        archi = 'GRU'

                    with open('expData.csv', 'a') as file:
                        line = opti + ' ' + archi + ' ' + str(maxEpochs) + ' ' + str(round(E_test, 5)) + ' ' + str(round(m.sqrt(S_test), 5)) + ' ' + str(round(E_train, 5)) + ' ' + str(round(m.sqrt(S_train), 5)) + ' ' + str(round(E_val, 5)) + ' ' + str(round(E_loss, 5))+ ' ' + str(count) + ' ' + str(learningRate) + ' ' + str(embeddin_size) + ' ' + str(k) + ' ' + str(lamb) + ' ' + str(dropout) + '\n'
                        file.write(line)


#krange = [500, 2500, 5000]
krange = [2500]
#lrrange = [1e-3, 1e-2, 1e-1]
lrrange = [1e-3]
#dropoutrange =  [0 0.1, 0.2, 0.3]
dropoutrange =  [0]
#lambrange = [1e-3, 1e-2, 1e-1]
lambrange = [1e-2]

#example experiment:
run_experiment(krange, lrrange, dropoutrange, lambrange, 1)            

