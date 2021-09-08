import numpy as np
from variables import *

def get_sample_weights(identifiers, train_dir, n_events):
    n_active = np.zeros((identifiers.shape[0], n_events))
    n_nonactive = np.zeros((identifiers.shape[0]))
    for i in range(identifiers.shape[0]):
        file = np.load(train_dir + 'y/' + identifiers[i] + '.npy')
        n_active[i] = np.sum(file[::,0::3], axis=0)

        n_nonactive[i] = (file.shape[0] * n_events) - np.sum(n_active[i])


    n_active = np.sum(n_active, axis=0)
    print(np.round(n_active/np.sum(n_active), 3), 'event class percentages')
    ratio = (1 / n_events) / n_active
    class_weight_ratio = ratio/np.sum(ratio)
    print(class_weight_ratio)
    class_weight_ratio = {0: class_weight_ratio[0], 1: class_weight_ratio[1], 2: class_weight_ratio[2]}

    active_nonactive_ratio = np.sum(n_nonactive)/(np.sum(n_nonactive)+np.sum(n_active))
    print(active_nonactive_ratio)
    active_nonactive_ratio = {0: 1 - active_nonactive_ratio, 1: active_nonactive_ratio}


    for i in range(identifiers.shape[0]):
        file = np.load(train_dir + 'y/' + identifiers[i] + '.npy')[::,0::3]
        sw = np.zeros(file.shape[0])
        for j in range(file.shape[0]):
            for k in range(n_events):
                if file[j,k] == 1:
                    sw[j] += class_weight_ratio[k] * active_nonactive_ratio[1]
                else: #0
                    sw[j] += class_weight_ratio[k] * active_nonactive_ratio[0]
        np.save(train_dir + 'sample_weights/' +identifiers[i]+'.npy', sw)



train_identifiers = np.load(train_identifiers_dir)
get_sample_weights(train_identifiers, train_dir, n_events=3)


# Event class percentages [0.087 0.077 0.836]
# Active/non-active ratio [0.17, 0.83,]