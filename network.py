import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
import numpy as np
from sliding_window import sliding_window_view
from keras import backend as K
from variables import params
# from keras.utils.vis_utils import plot_model #requires graphviz and pydot, pydotplus


# Convolutional layer
class CONVLayer(layers.Layer):
    def __init__(self, n_filters, kernel_size, stride_size):
        super(CONVLayer, self).__init__()
        self.kernel_size = kernel_size
        self.stride_size = stride_size

        self.conv = layers.Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=stride_size,
            padding='same',
            data_format="channels_last"
        )

        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x

# Recurrent layer
class GRULayer(layers.Layer):
    def __init__(self, n_units):
        super(GRULayer, self).__init__()
        self.n_units = n_units

        # Bidirectional GRU
        self.bidirectional = layers.Bidirectional(
            layer=layers.GRU(self.n_units, return_sequences=True,),
            backward_layer=layers.GRU(self.n_units, return_sequences=True, go_backwards=True,),
            merge_mode='concat',)

    def call(self, inputs, training=False):
        x = self.bidirectional(inputs)
        x = tf.nn.relu(x)
        x = layers.Dropout(0.25)(x)
        return x

# Convolutional block
class CONVBlock(layers.Layer):
    def __init__(self, params):
        super(CONVBlock, self).__init__()

        self.n_filters = params['filters'] # Convolutional filter size
        self.kernel_size = params['kernel_size']
        self.stride_size = (1,1)
        self.pool_size = [(1,5),(1,4),(1,2)]

        # Convolutional layers
        self.convlayer1 = CONVLayer(n_filters=self.n_filters, kernel_size=self.kernel_size, stride_size=self.stride_size,)
        self.convlayer2 = CONVLayer(n_filters=self.n_filters, kernel_size=self.kernel_size, stride_size=self.stride_size,)
        self.convlayer3 = CONVLayer(n_filters=self.n_filters, kernel_size=self.kernel_size, stride_size=self.stride_size,)

        # Max pooling layers
        self.poollayer1 = layers.MaxPool2D(pool_size=self.pool_size[0], strides=self.pool_size[0],)
        self.poollayer2 = layers.MaxPool2D(pool_size=self.pool_size[1], strides=self.pool_size[1],)
        self.poollayer3 = layers.MaxPool2D(pool_size=self.pool_size[2], strides=self.pool_size[2],)


    def call(self, inputs, training=False):

        x = self.convlayer1(inputs)
        x = self.poollayer1(x)
        x = layers.Dropout(0.25)(x)

        x = self.convlayer2(x)
        x = self.poollayer2(x)
        x = layers.Dropout(0.25)(x)

        x = self.convlayer3(x)
        x = self.poollayer3(x)
        x = layers.Dropout(0.25)(x)

        return x

# CRNN implementation
class CRNNModel(keras.Model):
    def __init__(self, params):
        super(CRNNModel, self).__init__()

        self.n_tfilters = 40 # triangular filters
        self.n_windows = params['frame_size'] # frame size

        self.n_units = params['filters'] # Recurrent layer unit size

        # Convolutional blocks for input streams
        self.CONVBlock1 = CONVBlock(params)
        self.CONVBlock2 = CONVBlock(params)
        self.CONVBlock3 = CONVBlock(params)

        self.gru_layer = GRULayer(n_units=self.n_units) # Recurrent (GRU) layer
        self.bn = layers.BatchNormalization() # Batch normalization
        self.fc = layers.Dense(self.n_units*2) # Residual fully connected layer
        self.out = layers.Dense(9, activation='sigmoid') # Flattened triplet output layer

    def call(self, inputs, training=False):

        # Inputs
        ar,lm,sdb = inputs

        # Convolutional blocks for each input
        ar = self.CONVBlock1(ar)
        lm = self.CONVBlock2(lm)
        sdb = self.CONVBlock3(sdb)

        # Remove size 1 dimension
        ar = tf.squeeze(ar, axis=2)
        lm = tf.squeeze(lm, axis=2)
        sdb = tf.squeeze(sdb, axis=2)

        # Concatenate feature maps
        x = tf.concat([ar,lm,sdb], axis=2)

        # Residual layer
        y = self.fc(x)
        y = tf.nn.relu(y)
        y = self.bn(y, training=training)
        y = layers.Dropout(0.25)(y)

        # Recurrent (GRU) layer
        x = self.gru_layer(x)

        # Concatenate residual and recurrent
        x = tf.concat([x,y],axis=2)

        # Output layer (flattened triplets)
        x = self.out(x)

        return x

    def model(self):

        # (n_sequences, n_windows, n_tfilters, n_channels)
        # Arousal (AR) stream
        x1 = keras.Input(shape=(self.n_windows, self.n_tfilters, 5), name='ar_input')
        # Limb movement (LM) stream
        x2 = keras.Input(shape=(self.n_windows, self.n_tfilters, 2), name='lm_input')
        # Sleep disordered breathing (SDB) stream
        x3 = keras.Input(shape=(self.n_windows, self.n_tfilters, 3), name='sdb_input')

        # Create model
        return keras.Model(inputs=[x1,x2,x3], outputs=self.call(inputs=[x1,x2,x3], training=True))


#  Multi-loss function, penalizing both event state classification and event boundary estimation
def loss_function(y_true, y_pred):
    # True triplet values
    y_true_y = y_true[::, ::, 0::3]
    y_true_p = y_true[::, ::, 1::3]
    y_true_q = y_true[::, ::, 2::3]

    # Predicted triplet values
    y_pred_y = y_pred[::, ::, 0::3]
    y_pred_p = y_pred[::, ::, 1::3]
    y_pred_q = y_pred[::, ::, 2::3]

    # Binary cross entropy
    y_pred_y_clipped = K.clip(y_pred_y, K.epsilon(), (1 - K.epsilon()))
    e_class = - y_true_y * K.log(y_pred_y_clipped+K.epsilon()) - (1-y_true_y) * K.log(1 - y_pred_y_clipped+K.epsilon())
    e_class = K.sum(e_class, axis=2)
    e_class = K.mean(e_class, axis=1)
    e_class = K.mean(e_class, axis=0) # Divide by batch

    # Distance loss
    e_dist = K.square(y_true_p - y_pred_p) + K.square(y_true_q - y_pred_q)
    e_dist = K.sum(e_dist, axis=2)
    e_dist = K.mean(e_dist, axis=1)
    e_dist = K.mean(e_dist, axis=0) # Divide by batch

    # Confidence loss
    intersection = K.minimum(y_true_p, y_pred_p) + K.minimum(y_true_q, y_pred_q)
    union = K.maximum(y_true_p, y_pred_p) + K.maximum(y_true_q, y_pred_q)
    e_conf = K.square(y_true_y - (intersection/union))
    e_conf = K.sum(e_conf, axis=2)
    e_conf = K.mean(e_conf, axis=1)
    e_conf = K.mean(e_conf, axis=0) # Divide by batch

    # Overall loss
    loss = e_class + e_dist + e_conf
    return loss

# Error rate 'ER' (using keras backend)
def error_rate_metric(y_true, y_pred):
    threshold = 0.5  # Training threshold 0.5
    y_true_y = y_true[::, ::, 0::3]
    y_pred_y = y_pred[::, ::, 0::3]
    y_pred_y = K.cast(K.greater(K.clip(y_pred_y, 0, 1), threshold), K.floatx())

    # Segment based ER
    # [n_seqs * n_frames, 3]
    data_shape = keras.backend.shape(y_true_y)
    y_true_y = keras.backend.reshape(y_true_y, (data_shape[0]*data_shape[1],data_shape[2]))
    y_pred_y = keras.backend.reshape(y_pred_y, (data_shape[0]*data_shape[1],data_shape[2]))

    actives_N = K.sum(y_true_y, axis=1) # Number of active events in groundtruths
    false_negatives = K.sum(K.clip(y_true_y * (1 - y_pred_y), 0, 1), axis=1) # False negatives of each frame
    false_positives = K.sum(K.clip((1 - y_true_y) * y_pred_y, 0, 1), axis=1) # False positives of each frame

    zero_tensor = K.zeros_like(actives_N)

    substitutions_S = K.minimum(false_negatives, false_positives) # Substitutions at each frame
    deletions_D = K.maximum(zero_tensor, (false_negatives - false_positives)) # Deletions at each frame
    insertions_I = K.maximum(zero_tensor, (false_positives - false_negatives)) # Insertions at each frame


    # Conditional statement in case of 0 active events to avoid zero error.
    # Formula : (S+D+I)/(N)
    error_rate = K.switch(
        K.greater(K.sum(actives_N), 0),
        (K.sum(substitutions_S) + K.sum(deletions_D) + K.sum(insertions_I)) / (K.sum(actives_N)),
        1.0,
    )

    return error_rate

# Precision (using keras backend)
def precision_metric(y_true, y_pred):
    threshold = 0.5  # Training threshold 0.5
    y_true_y = y_true[::, ::, 0::3]
    y_pred_y = y_pred[::, ::, 0::3]
    y_pred_y = K.cast(K.greater(K.clip(y_pred_y, 0, 1), threshold), K.floatx())

    true_positives = K.sum(K.clip(y_true_y * y_pred_y, 0, 1))
    false_negatives = K.sum(K.clip(y_true_y * (1-y_pred_y), 0, 1))
    false_positives = K.sum(K.clip((1-y_true_y) * y_pred_y, 0, 1))
    true_negatives = K.sum(K.clip((1 - y_true_y) * (1-y_pred_y), 0, 1))

    precision = true_positives / (true_positives + false_positives + K.epsilon())
    return precision

# Recall (using keras backend)
def recall_metric(y_true, y_pred):
    threshold = 0.5 #Training threshold 0.5
    y_true_y = y_true[::, ::, 0::3]
    y_pred_y = y_pred[::, ::, 0::3]
    y_pred_y = K.cast(K.greater(K.clip(y_pred_y, 0, 1), threshold), K.floatx())

    true_positives = K.sum(K.clip(y_true_y * y_pred_y, 0, 1))
    false_negatives = K.sum(K.clip(y_true_y * (1-y_pred_y), 0, 1))
    false_positives = K.sum(K.clip((1-y_true_y) * y_pred_y, 0, 1))
    true_negatives = K.sum(K.clip((1 - y_true_y) * (1-y_pred_y), 0, 1))

    recall = true_positives / (true_positives + false_negatives + K.epsilon())
    return recall

# F1-score (using keras backend)
def f1_metric(y_true, y_pred):
    threshold = 0.5  # Training threshold 0.5
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (recall+precision+K.epsilon()))
    return f1

# The data generator loads and prepares each batch for the model input
def dataGenerator(identifiers, batch_size, file_dir, params, training):

    def compute_seqs(signal, seq_window_length, seq_stride_length):
        return sliding_window_view(
            x=signal,
            window_shape=seq_window_length,
            axis=0,
        )[::seq_stride_length, ::]

    # True until all batches are processed
    while True:
        # Randomly shuffle identifiers
        np.random.shuffle(identifiers)

        # Iterate through each identifier
        for id in identifiers:

            # Load x and y data and convert them into sequences
            x_data = np.load(file_dir + 'x/' + id + '.npy')
            y_data = np.load(file_dir + 'y/' + id + '.npy')

            x_data = compute_seqs(x_data, params['frame_size'], params['frame_size'])
            x_data = np.swapaxes(x_data, 2, 3)
            x_data = np.swapaxes(x_data, 1, 2)

            y_data = compute_seqs(y_data, params['frame_size'], params['frame_size'])
            y_data = np.swapaxes(y_data, 1, 2)

            # If it is a training datagenerator, load training sample weights
            if training:
                sample_weights = np.load(file_dir + 'sample_weights/' + id + '.npy')

            # Split the data into batches pre-defined by the batch-size
            for j in range(0, x_data.shape[0], batch_size):
                x = x_data[j:j + batch_size]

                ar = x[:,:,:,0:5]
                lm = x[:,:,:,5:7]
                sdb = x[:,:,:,7:10]
                x = [ar,lm,sdb]

                y = y_data[j:j + batch_size]

                if training:
                    sw = sample_weights[j:j + batch_size]
                    yield x, y, sw
                else:
                    yield x, y

# Compute the number of batches in the corresponding set
def get_nbatches(identifiers, batch_size, file_dir, params):
    n_seqs = np.zeros((identifiers.shape[0]))
    for i in range(identifiers.shape[0]):
        count = np.load(file_dir + 'y/' + identifiers[i] + '.npy').shape[0]
        count = np.floor(count / params['frame_size'])
        n_seqs[i] += count

    return np.sum(np.ceil(n_seqs/batch_size))

# model = CRNNModel(params).model()
# model.summary()
# plot_model(model, show_shapes=True, show_layer_names=True, dpi=100)