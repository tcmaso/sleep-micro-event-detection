from network import *
from variables import *
import numpy as np

batch_size = 128
# Parameters to search
params_list = {'frame_size':[16,32,64,128],
          'kernel_size':[(3,3),(5,5),(11,11),(1,5),(5,1)],
          'filters': [96,256]}

# Iterate through each parameter combination
for frame_size in params_list['frame_size']:
    for kernel_size in params_list['kernel_size']:
        for filters in params_list['filters']:

            # Current parameter configuration
            params = {'frame_size': frame_size,
                      'kernel_size': kernel_size,
                      'filters': filters}
            print(params)

            # Initate model with current parameter configuration
            model = CRNNModel(params).model()

            # Compile model
            model.compile(
            optimizer = keras.optimizers.Adam(lr=0.0001),
            loss = loss_function,
            metrics = [recall_metric, precision_metric, f1_metric]
            )

            # Load training and validation identifiers
            train_identifiers = np.load(train_identifiers_dir)
            val_identifiers = np.load(val_identifiers_dir)

            # Create training and validation data generators
            train_generator = dataGenerator(train_identifiers, batch_size, train_dir, params, training=True)
            val_generator = dataGenerator(val_identifiers, batch_size, val_dir, params, training=False)

            # Number of steps (batches) for training and validation
            train_nbatches = get_nbatches(train_identifiers, batch_size, train_dir, params)
            val_nbatches = get_nbatches(val_identifiers, batch_size, val_dir, params)

            # Commence training
            history = model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_nbatches,
                                validation_data=val_generator,
                                validation_steps=val_nbatches,
                                epochs=1,
                                verbose=1,
                                )

            print(history.history['val_f1_metric'])

