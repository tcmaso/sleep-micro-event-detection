import os
import numpy as np
from variables import *
from network import *
from matplotlib import pyplot as plt

batch_size = 128 # 128 sequences per step
# params = {'frame_size': 64, 'kernel_size':(5,5), 'filters': 256}

# Initialise model
model = CRNNModel(params).model()
model.summary()

# Load existing model or create new model
if os.path.isfile(model_dir + '/weights/model_weights.index'):
    model.load_weights(model_dir + '/weights/model_weights')
else:
    model.save(model_dir)
    model.save_weights(model_dir + '/weights/model_weights')

# Compile model for training
model.compile(
optimizer = keras.optimizers.Adam(lr=0.0001),
loss = loss_function,
metrics = [recall_metric, precision_metric, f1_metric, error_rate_metric]
)

# Load training and validation identifiers
train_identifiers = np.load(train_identifiers_dir)
val_identifiers = np.load(val_identifiers_dir)

# Number of steps (batches) for training and validation
train_nbatches = get_nbatches(train_identifiers, batch_size, train_dir, params)
val_nbatches = get_nbatches(val_identifiers, batch_size, val_dir, params)

# Create training and validation data generators
train_generator = dataGenerator(train_identifiers, batch_size, train_dir, params, training=True)
val_generator = dataGenerator(val_identifiers, batch_size, val_dir, params, training=False)

# Commence training
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_nbatches,
                    validation_data=val_generator,
                    validation_steps=val_nbatches,
                    epochs=1,
                    verbose=1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_f1_metric', mode='max', restore_best_weights=True)]
                    )

model.save_weights(model_dir + '/weights/model_weights')

train_f1 = history.history['f1_metric']
val_f1 = history.history['val_f1_metric']
plt.plot(train_f1)
plt.plot(val_f1)
plt.title("Training f1 vs Validation f1")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.legend(["Training", "Validation"])
plt.show()
