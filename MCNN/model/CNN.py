import numpy as np
import keras
import tensorflow as tf

# Model input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
# 
#     V: num_of_vertices
#     T: num_of_timesteps
#     F: num_of_features
#
# Model output: (*, 5)


from keras import models,layers

def CNN(opt='adam'):
    n_context,n_Channel,n_Channel_m, n_Time = 5,8,1,3000

    ###############################################################################
    EEG_layer = layers.Input(shape=(n_context, n_Channel, n_Time, 1))
    EMG_layer = layers.Input(shape=(n_context, n_Channel_m, n_Time, 1))

    ###############################################################################
    EEG_in = layers.Input(shape=(n_Channel, n_Time, 1))
    Conv0_out = layers.Conv2D(
        filters = n_Channel,
        kernel_size = (n_Channel, 1), 
        strides=(1,1),
        use_bias=False,
        activation='linear')(EEG_in)

    Permute_out = layers.Permute((3,2,1))(Conv0_out)

    Conv1_out = layers.Conv2D(
        filters = 8,
        kernel_size = (1,64), 
        strides=(1,1),
        padding = 'same',
        activation='relu')(Permute_out)

    MaxP1_out = layers.MaxPooling2D(
        pool_size = (1,16), 
        strides=(1,16))(Conv1_out)

    Conv2_out = layers.Conv2D(
        filters = 8,
        kernel_size = (1,64), 
        strides=(1,1),
        padding = 'same',
        activation='relu')(MaxP1_out)

    MaxP2_out = layers.MaxPooling2D(
        pool_size = (1,16), 
        strides=(1,16))(Conv2_out)

    Flat_out = layers.Flatten()(MaxP2_out)

    Drop_out = layers.Dropout(0.5)(Flat_out)

    EEG_FE = models.Model(EEG_in,Drop_out)

    ###############################################################################
    EMG_in = layers.Input(shape=(n_Channel_m, n_Time, 1))
    Conv0_out_m = layers.Conv2D(
        filters = n_Channel_m,
        kernel_size = (n_Channel_m, 1), 
        strides=(1,1),
        use_bias=False,
        activation='linear')(EMG_in)

    Permute_out_m = layers.Permute((3,2,1))(Conv0_out_m)

    Conv1_out_m = layers.Conv2D(
        filters = 8,
        kernel_size = (1,64), 
        strides=(1,1),
        padding = 'same',
        activation='relu')(Permute_out_m)

    MaxP1_out_m = layers.MaxPooling2D(
        pool_size = (1,16), 
        strides=(1,16))(Conv1_out_m)

    Conv2_out_m = layers.Conv2D(
        filters = 8,
        kernel_size = (1,64), 
        strides=(1,1),
        padding = 'same',
        activation='relu')(MaxP1_out_m)

    MaxP2_out_m = layers.MaxPooling2D(
        pool_size = (1,16), 
        strides=(1,16))(Conv2_out_m)

    Flat_out_m = layers.Flatten()(MaxP2_out_m)

    Drop_out_m = layers.Dropout(0.5)(Flat_out_m)

    EMG_FE = models.Model(EMG_in,Drop_out_m)

    EEG_out = layers.TimeDistributed(EEG_FE)(EEG_layer)
    EMG_out = layers.TimeDistributed(EMG_FE)(EMG_layer)

    FE_out = layers.concatenate([EEG_out,EMG_out])

    FE_out = layers.Flatten()(FE_out)
    softmax = layers.Dense(5,activation='softmax')(FE_out)

    model = models.Model(inputs = [EEG_layer,EMG_layer], outputs = softmax)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc']
    )
    return model