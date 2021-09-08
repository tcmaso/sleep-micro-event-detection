import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from keras import backend as K
from sliding_window import sliding_window_view
import numpy as np
from matplotlib import pyplot as plt
from network import CRNNModel
from variables import *

# Computes sequences using sliding window
def compute_seqs(signal, seq_window_length, seq_stride_length):
    return sliding_window_view(
        x=signal,
        window_shape=seq_window_length,
        axis=0,
    )[::seq_stride_length, ::]

# Recall (Numpy version)
def compute_recall(y_true, y_pred, threshold, event_type=None):
    if event_type==None:
        y_true_y = y_true
        y_pred_y = (y_pred > threshold).astype('int')
    else:
        y_true_y = y_true[::, ::, event_type]
        y_pred_y = y_pred[::, ::, event_type]
        y_pred_y = (y_pred_y > threshold[event_type]).astype('int')

    true_positives = np.sum(y_true_y * y_pred_y)
    false_negatives = np.sum(y_true_y * (1 - y_pred_y))
    false_positives = np.sum((1 - y_true_y) * y_pred_y)
    true_negatives = np.sum((1 - y_true_y) * (1 - y_pred_y))

    recall = true_positives / (true_positives + false_negatives + np.finfo(np.float32).eps)
    return recall

# Precision (Numpy version)
def compute_precision(y_true, y_pred, threshold, event_type=None):
    if event_type==None:
        y_true_y = y_true
        y_pred_y = (y_pred > threshold).astype('int')
    else:
        y_true_y = y_true[::, ::, event_type]
        y_pred_y = y_pred[::, ::, event_type]
        y_pred_y = (y_pred_y > threshold[event_type]).astype('int')

    true_positives = np.sum(y_true_y * y_pred_y)
    false_negatives = np.sum(y_true_y * (1 - y_pred_y))
    false_positives = np.sum((1 - y_true_y) * y_pred_y)
    true_negatives = np.sum((1 - y_true_y) * (1 - y_pred_y))

    precision = true_positives / (true_positives + false_positives + np.finfo(np.float32).eps)

    return precision

# F1-score (Numpy version)
def compute_f1(y_true, y_pred, threshold, event_type=None):
    recall = compute_recall(y_true, y_pred, threshold, event_type)
    precision = compute_precision(y_true, y_pred, threshold, event_type)
    f1 = 2 * ((precision * recall) / (recall + precision+np.finfo(np.float32).eps))
    return f1

# Error rate 'ER' (Numpy version)
def compute_er(y_true, y_pred, threshold, event_type=None):
    if event_type==None:
        y_true_y = y_true
        y_pred_y = (y_pred > threshold).astype('int')
    else:
        y_true_y = y_true[::, ::, event_type]
        y_pred_y = y_pred[::, ::, event_type]
        y_pred_y = (y_pred_y > threshold[event_type]).astype('int')


    # Segment based ER
    # [n_seqs * n_frames, 3]
    data_shape = y_true_y.shape
    if event_type==None:
        y_true_y = np.reshape(y_true_y, (data_shape[0] * data_shape[1], data_shape[2]))
        y_pred_y = np.reshape(y_pred_y, (data_shape[0] * data_shape[1], data_shape[2]))

        actives_N = np.sum(y_true_y, axis=1)  # Number of active events of each frame in groundtruths
        false_negatives = np.sum(y_true_y * (1 - y_pred_y), axis=1)  # False negatives of each frame
        false_positives = np.sum((1 - y_true_y) * y_pred_y, axis=1)  # False positives of each frame
    else:
        y_true_y = np.reshape(y_true_y, (data_shape[0] * data_shape[1]))
        y_pred_y = np.reshape(y_pred_y, (data_shape[0] * data_shape[1]))

        actives_N = y_true_y  # Number of active events in groundtruths
        false_negatives = y_true_y * (1 - y_pred_y)  # False negatives of each frame
        false_positives = (1 - y_true_y) * y_pred_y  # False positives of each frame

    zero_tensor = np.zeros(actives_N.shape)
    substitutions_S = np.minimum(false_negatives, false_positives)  # Substitutions at each frame
    deletions_D = np.maximum(zero_tensor, (false_negatives - false_positives))  # Deletions at each frame
    insertions_I = np.maximum(zero_tensor, (false_positives - false_negatives))  # Insertions at each frame

    error_rate = (np.sum(substitutions_S) + np.sum(deletions_D) + np.sum(insertions_I)) / np.sum(actives_N)
    return error_rate



# Compute confidence scores using inference scheme
def compute_confidence_scores(identifiers, data_dir, model_dir, project_dir, conf_dir, params, max_event_lengths):

    # Initialise model and load trained weights
    model = CRNNModel(params).model()
    model.load_weights(model_dir + '/weights/model_weights')

    # Iterate through all identifiers
    for i in range(identifiers.shape[0]):
        # Load x and y data and compute sequences for model input
        x_data = np.load(data_dir + 'x/' + identifiers[i] + '.npy')
        y_data = np.load(data_dir + 'y/' + identifiers[i] + '.npy')

        x_data = compute_seqs(x_data, params['frame_size'], params['frame_size'])
        x_data = np.swapaxes(x_data, 2, 3)
        x_data= np.swapaxes(x_data, 1, 2)

        ar = x_data[:, :, :, 0:5]
        lm = x_data[:, :, :, 5:7]
        sdb = x_data[:, :, :, 7:10]
        x_data = [ar, lm, sdb]


        total_n_frames = y_data.shape[0]
        y_data = compute_seqs(y_data, params['frame_size'], params['frame_size'])
        y_data = np.swapaxes(y_data, 1, 2)

        # Predicted output using current input
        y_hat_data = model.predict(x_data)

        n_seqs = y_hat_data.shape[0]
        n_frames = y_hat_data.shape[1]
        n_events = int(y_hat_data.shape[2] / 3)

        f_n = np.zeros((n_seqs * n_frames, n_events))

        # Inference scheme
        # Iterates through all sequences (n)
        for s in range(n_seqs):

            # Iterate through all temporal positions of current frame (n*)
            for n in range(s * n_frames, (s * n_frames) + n_frames):

                if n >= total_n_frames:
                    break

                for t in range(n_frames):

                    for k in range(n_events):

                        # Predicted event state, onset and offset for current sequence, frame and event
                        y_hat_t, p_hat_t, q_hat_t = y_hat_data[s, t, (k * 3): (k * 3 + 3)]

                        # Denormalize
                        p_hat_t *= max_event_lengths[k]
                        q_hat_t *= max_event_lengths[k]

                        # Ensure current temporal position (n*) is in region of interest
                        if (n) >= (s * n_frames + t - p_hat_t) and (n) <= (s * n_frames + t + q_hat_t):
                            roi = 1
                        else:
                            roi = 0

                        f_n[n, k] += (y_hat_t * roi)

        # Save temporal seqeunce of segments (confidence scores) for current identifier
        np.save(conf_dir + identifiers[i] + '.npy', f_n)
        print(identifiers[i], 'processed')
        print(f_n.shape)


# Compute max confidence scores for normalization
def compute_max_confidence_scores(identifiers, conf_dir, set_type):
    max_conf_scores = np.array([0,0,0])
    for i in range(identifiers.shape[0]):
        conf_data = np.load(conf_dir + identifiers[i] + '.npy')
        current_max = np.max(conf_data, axis=(0))
        max_conf_scores = np.maximum(max_conf_scores,current_max)

    np.save(project_dir + set_type + '_max_conf_scores' + '.npy', max_conf_scores)
    return max_conf_scores


# Cross validation to compute f1-scores for each event type for all combinations of alpha and beta threshold
def cv_scores(identifiers, data_dir, model_dir, project_dir, conf_dir, params, a_thresholds, b_thresholds, max_conf_scores):
    # Initialise model and load trained weights
    model = CRNNModel(params).model()
    model.load_weights(model_dir + '/weights/model_weights')
    print(identifiers.shape)

    # Vectors [f1_ar, f1_lm, f1_sdb]
    cv_scores = np.zeros((identifiers.shape[0], a_thresholds.shape[0], b_thresholds.shape[0], 3))
    np.save(project_dir + 'cv_scores.npy', cv_scores)

    # Iterate through all identifiers
    for i in range(identifiers.shape[0]):

        # Load x and y data and compute sequences for model input
        x_data = np.load(data_dir + 'x/' + identifiers[i] + '.npy')
        x_data = compute_seqs(x_data, params['frame_size'], params['frame_size'])
        x_data = np.swapaxes(x_data, 2, 3)
        x_data= np.swapaxes(x_data, 1, 2)

        ar = x_data[:, :, :, 0:5]
        lm = x_data[:, :, :, 5:7]
        sdb = x_data[:, :, :, 7:10]
        x_data = [ar, lm, sdb]

        y_data = np.load(data_dir + 'y/' + identifiers[i] + '.npy')
        y_data = compute_seqs(y_data, params['frame_size'], params['frame_size'])
        y_data = np.swapaxes(y_data, 1, 2)
        y_data = y_data[::, ::, 0::3]

        # Predicted output using current input
        y_hat_data = model.predict(x_data)
        # Predicted y event states
        y_hat_data = y_hat_data[::, ::, 0::3]

        # Load confidence scores
        y_conf_data = np.load(conf_dir + identifiers[i] + '.npy')
        # normalize confidence scores
        y_conf_data = y_conf_data / max_conf_scores
        # convert confidence scores to sequences
        y_conf_data = compute_seqs(y_conf_data, params['frame_size'], params['frame_size'])
        y_conf_data = np.swapaxes(y_conf_data, 1, 2)

        # Cross validation for optimal alpha and beta thresholds
        cv_score = np.zeros((a_thresholds.shape[0], b_thresholds.shape[0], 3))
        for a in range(a_thresholds.shape[0]):

            a_threshold = [a_thresholds[a]] * 3

            # Apply alpha event state threshold to confidence scores using indicator function
            y_conf_data_alpha = y_conf_data * (y_hat_data > a_threshold).astype('int')

            for b in range(b_thresholds.shape[0]):

                b_threshold = [b_thresholds[b]] * 3

                # Compute f1-scores for the AR, LM and SDB events for current alpha and beta threshold
                f1_ar = compute_f1(y_data, y_conf_data_alpha, b_threshold, event_type=0)
                f1_lm = compute_f1(y_data, y_conf_data_alpha, b_threshold, event_type=1)
                f1_sdb = compute_f1(y_data, y_conf_data_alpha, b_threshold, event_type=2)

                cv_score[a,b] = [f1_ar, f1_lm, f1_sdb]

        # Load and save alpha and beta threshold results for current identifier
        cv_scores = np.load(project_dir + 'cv_scores.npy')
        cv_scores[i] = cv_score
        np.save(project_dir + 'cv_scores.npy', cv_scores)
        print(identifiers[i], 'processed')

    return cv_scores

# Obtain best alpha and beta threshold
def best_cv_thresholds(project_dir, a_thresholds, b_thresholds):
    cv_scores = np.load(project_dir + 'cv_scores.npy')
    cv_scores = np.mean(cv_scores, axis=0)

    best_a_thresholds = np.zeros(cv_scores.shape[2])
    best_b_thresholds = np.zeros(cv_scores.shape[2])

    for i in range(best_a_thresholds.shape[0]):

        indices = np.where(cv_scores[:,:,i] == np.max(cv_scores[:,:,i]))
        best_a_thresholds[i] = a_thresholds[indices[0][0]]
        best_b_thresholds[i] = b_thresholds[indices[1][0]]

    return best_a_thresholds, best_b_thresholds



def evaluation_metrics(identifiers, data_dir, model_dir, project_dir, conf_dir, params, best_a_thresholds, best_b_thresholds, max_conf_scores, set_type):
    # Initialise model and load trained weights
    model = CRNNModel(params).model()
    model.load_weights(model_dir + '/weights/model_weights')

    # vectors [Precision, recall, f1, er]
    evaluation_scores = np.zeros((identifiers.shape[0], 4, 4))
    np.save(project_dir + set_type + '_evaluation_scores.npy', evaluation_scores)

    # Iterate through all identifiers
    for i in range(identifiers.shape[0]):
        # Load x and y data and compute sequences for model input
        x_data = np.load(data_dir + 'x/' + identifiers[i] + '.npy')
        x_data = compute_seqs(x_data, params['frame_size'], params['frame_size'])
        x_data = np.swapaxes(x_data, 2, 3)
        x_data = np.swapaxes(x_data, 1, 2)

        ar = x_data[:, :, :, 0:5]
        lm = x_data[:, :, :, 5:7]
        sdb = x_data[:, :, :, 7:10]
        x_data = [ar, lm, sdb]

        y_data = np.load(data_dir + 'y/' + identifiers[i] + '.npy')
        y_data = compute_seqs(y_data, params['frame_size'], params['frame_size'])
        y_data = np.swapaxes(y_data, 1, 2)
        y_data = y_data[::, ::, 0::3]

        # Predicted output using current input
        y_hat_data = model.predict(x_data)
        # Predicted y event states
        y_hat_data = y_hat_data[::, ::, 0::3]

        # Load confidence scores
        y_conf_data = np.load(conf_dir + identifiers[i] + '.npy')
        # normalize confidence scores
        y_conf_data = y_conf_data / max_conf_scores
        # convert confidence scores to sequences
        y_conf_data = compute_seqs(y_conf_data, params['frame_size'], params['frame_size'])
        y_conf_data = np.swapaxes(y_conf_data, 1, 2)

        # Apply best alpha event state threshold to confidence scores using indicator function
        y_conf_data_alpha = y_conf_data * (y_hat_data >= best_a_thresholds).astype('int')

        # Compute evaluation metrics for all events and overall performance
        evaluation_score = np.zeros((4,4))
        #======
        precision_overall = compute_precision(y_data, y_conf_data_alpha, best_b_thresholds)
        recall_overall = compute_recall(y_data, y_conf_data_alpha, best_b_thresholds)
        f1_overall = compute_f1(y_data, y_conf_data_alpha, best_b_thresholds)
        er_overall = compute_er(y_data, y_conf_data_alpha, best_b_thresholds)
        evaluation_score[0] = [precision_overall, recall_overall, f1_overall, er_overall]

        precision_ar = compute_precision(y_data, y_conf_data_alpha, best_b_thresholds, event_type=0)
        recall_ar = compute_recall(y_data, y_conf_data_alpha, best_b_thresholds, event_type=0)
        f1_ar = compute_f1(y_data, y_conf_data_alpha, best_b_thresholds, event_type=0)
        er_ar = compute_er(y_data, y_conf_data_alpha, best_b_thresholds, event_type=0)
        evaluation_score[1] = [precision_ar, recall_ar, f1_ar, er_ar]

        precision_lm = compute_precision(y_data, y_conf_data_alpha, best_b_thresholds, event_type=1)
        recall_lm = compute_recall(y_data, y_conf_data_alpha, best_b_thresholds, event_type=1)
        f1_lm = compute_f1(y_data, y_conf_data_alpha, best_b_thresholds, event_type=1)
        er_lm = compute_er(y_data, y_conf_data_alpha, best_b_thresholds, event_type=1)
        evaluation_score[2] = [precision_lm, recall_lm, f1_lm, er_lm]

        precision_sdb = compute_precision(y_data, y_conf_data_alpha, best_b_thresholds, event_type=2)
        recall_sdb = compute_recall(y_data, y_conf_data_alpha, best_b_thresholds, event_type=2)
        f1_sdb = compute_f1(y_data, y_conf_data_alpha, best_b_thresholds, event_type=2)
        er_sdb = compute_er(y_data, y_conf_data_alpha, best_b_thresholds, event_type=2)
        evaluation_score[3] = [precision_sdb, recall_sdb, f1_sdb, er_sdb]

        evaluation_scores = np.load(project_dir + set_type + '_evaluation_scores.npy')

        evaluation_scores[i] = evaluation_score
        np.save(project_dir + set_type + '_evaluation_scores.npy', evaluation_scores)
        print(identifiers[i], 'processed')


def plot_confidence(file_id):

    # Confidence score unthresholded
    y_conf_data = np.load(test_conf_dir + test_identifiers[file_id] + '.npy')#[12000:17000] # 838
    y_conf_data = y_conf_data / val_max_conf_scores

    # Obtain predicted event states
    x_data = np.load(test_dir + 'x/' + test_identifiers[file_id] + '.npy')
    x_data = compute_seqs(x_data, params['frame_size'], params['frame_size'])
    x_data = np.swapaxes(x_data, 2, 3)
    x_data = np.swapaxes(x_data, 1, 2)
    ar = x_data[:, :, :, 0:5]
    lm = x_data[:, :, :, 5:7]
    sdb = x_data[:, :, :, 7:10]
    x_data = [ar, lm, sdb]

    model = CRNNModel(params).model()
    model.load_weights(model_dir + '/weights/model_weights')

    # Predicted y event states
    y_hat_data = model.predict(x_data)
    y_hat_data = y_hat_data[::, ::, 0::3]
    y_hat_data = np.reshape(y_hat_data, (y_hat_data.shape[0]*y_hat_data.shape[1],y_hat_data.shape[2]))#[12000:17000] # 838

    # Apply alpha event state threshold to confidence scores using indicator function
    y_conf_data_alpha = y_conf_data * (y_hat_data >= best_a_thresholds).astype('int')

    # Beta threshold applied to confidence score for predicted event states
    y_hat_data = (y_conf_data_alpha > best_b_thresholds).astype('int')
    events_predicted = np.array([np.arange(0,y_hat_data.shape[0]),np.arange(0,y_hat_data.shape[0]),np.arange(0,y_hat_data.shape[0])])
    events_predicted = events_predicted * y_hat_data.T

    ar_events_predicted = events_predicted[0,:][events_predicted[0,:]>0]
    lm_events_predicted = events_predicted[1,:][events_predicted[1,:]>0]
    sdb_events_predicted = events_predicted[2,:][events_predicted[2,:]>0]


    # Groundtruth event states
    y_data = np.load(test_dir + 'y/' + test_identifiers[file_id] + '.npy')[:,0::3]#[12000:17000] # 838
    events_groundtruth = np.array([np.arange(0,y_data.shape[0]),np.arange(0,y_data.shape[0]),np.arange(0,y_data.shape[0])])
    events_groundtruth = events_groundtruth * y_data.T
    ar_events_groundtruth = events_groundtruth[0,:][events_groundtruth[0,:]>0]
    lm_events_groundtruth = events_groundtruth[1,:][events_groundtruth[1,:]>0]
    sdb_events_groundtruth = events_groundtruth[2,:][events_groundtruth[2,:]>0]

    fig = plt.figure(figsize=(8, 8), dpi=600)
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    # AR, LM and SDB confidence scores
    ax1.plot(y_conf_data_alpha[:,0] , 'r')
    ax1.set_yticks([0,0.5,1.0])
    ax1.axhline(best_b_thresholds[0], ls='--', c='k')
    ax1.set_ylabel('Confidence score')

    ax2.plot(y_conf_data_alpha[:,1], 'g')
    ax2.set_yticks([0,0.5,1.0])
    ax2.axhline(best_b_thresholds[1], ls='--', c='k')
    ax2.set_ylabel('Confidence score')

    ax3.plot(y_conf_data_alpha[:,2], 'b')
    ax3.set_yticks([0,0.5,1.0])
    ax3.axhline(best_b_thresholds[2], ls='--', c='k')
    ax3.set_ylabel('Confidence score')

    # Predicted and groundtruth event states
    ax4.eventplot(ar_events_predicted, lineoffsets=[2.7],
                        linelengths=[0.5], colors='r')

    ax4.eventplot(ar_events_groundtruth, lineoffsets=[2.2],
                        linelengths=[0.5], colors='palevioletred')

    ax4.eventplot(lm_events_predicted, lineoffsets=[1.6],
                        linelengths=[0.5], colors='g')

    ax4.eventplot(lm_events_groundtruth, lineoffsets=[1.1],
                        linelengths=[0.5], colors='palegreen')

    ax4.eventplot(sdb_events_predicted, lineoffsets=[0.5],
                        linelengths=[0.5], colors='b')

    ax4.eventplot(sdb_events_groundtruth, lineoffsets=[0],
                        linelengths=[0.5], colors='paleturquoise')
    ax4.set_yticklabels([])
    ax4.set_yticks([])
    ax4.set_ylabel('Event active states', labelpad=15)
    ax4.legend(['AR Prediction','AR Groundtruth','LM Prediction','LM Groundtruth','SDB Prediction','SDB Groundtruth'],bbox_to_anchor=(0, -0.6, 1, -0.6),  mode="expand", ncol=3, loc='lower center')

    plt.savefig(project_dir + 'confidence_plot.png', dpi=1200)
    plt.show()



#################################################################


params = {'frame_size': 64, 'kernel_size':(5,5), 'filters': 256}
thresholds = np.arange(0, 1, 0.01)

train_identifiers = np.load(train_identifiers_dir)
val_identifiers = np.load(val_identifiers_dir)
test_identifiers = np.load(test_identifiers_dir)

# Max events in training data (for denormalization) # [85, 34, 13261]
max_event_lengths = np.load(project_dir + 'training_max_event_lengths.npy')


# Compute validation confidence scores
if os.listdir(val_inference_conf_dir) == []:
    compute_confidence_scores(val_identifiers, val_inference_dir, model_dir, project_dir, val_inference_conf_dir, params,
                          max_event_lengths)


# Compute validation max confidence scores
if not os.path.isfile(project_dir + 'val_max_conf_scores.npy'):
    val_max_conf_scores = compute_max_confidence_scores(val_identifiers, val_inference_conf_dir, set_type='val')
else:
    val_max_conf_scores = np.load(project_dir + 'val_max_conf_scores' + '.npy')
    # [44.28404713, 10.60794538, 63.28111494]


# Alpha, beta threshold parameters for cross-validation
a_thresholds = np.arange(0, 1.01, 0.01)
b_thresholds = np.arange(0, 1.01, 0.01)


# Cross-validation to find optimal alpha, beta parameters that maximize f1-score
if not os.path.isfile(project_dir + 'cv_scores.npy'):
    cv_scores(val_identifiers, val_inference_dir, model_dir, project_dir, val_inference_conf_dir, params, a_thresholds, b_thresholds, val_max_conf_scores)
else:
    cv_scores = np.load(project_dir + 'cv_scores.npy')


# Obtain best alpha and beta thresholds values from the cross-validation
best_a_thresholds, best_b_thresholds = best_cv_thresholds(project_dir, a_thresholds, b_thresholds)
#[0.29 0.36 0.  ] [0.01 0.01 0.33]
print(best_a_thresholds, best_b_thresholds)


# Compute test confidence scores
if os.listdir(test_conf_dir) == []:
    compute_confidence_scores(test_identifiers, test_dir, model_dir, project_dir, test_conf_dir, params,
                              max_event_lengths)


# Compute evluation metrics for test set
if not os.path.isfile(project_dir + 'test_evaluation_scores.npy'):
    print('x')
    # evaluation_metrics(test_identifiers, test_dir, model_dir, project_dir, test_conf_dir, params, best_a_thresholds, best_b_thresholds, val_max_conf_scores, set_type='test')
else:
    test_evaluation_scores =  np.load(project_dir + 'test_evaluation_scores.npy')


# Evaluation metric scores
print('Evaluation metric scores:')
test_evaluation_scores = np.load(project_dir + 'test_evaluation_scores.npy')
lm_ermean = test_evaluation_scores[:,2,3]
lm_ermean = lm_ermean[lm_ermean!= np.inf]
lm_ermean = np.mean(lm_ermean, axis=0)
test_evaluation_scores = np.mean(test_evaluation_scores, axis=0)
test_evaluation_scores[2,3] = lm_ermean
print(test_evaluation_scores)

# Evaluation metric standard deviations
print('Standard deviations:')
test_evaluation_scores = np.load(project_dir + 'test_evaluation_scores.npy')
lm_ermean = test_evaluation_scores[:,2,3]
lm_ermean = lm_ermean[lm_ermean!= np.inf]
lm_ermean = np.std(lm_ermean, axis=0)
test_evaluation_scores = np.std(test_evaluation_scores, axis=0)
test_evaluation_scores[2,3] = lm_ermean
print(test_evaluation_scores)

# Precision boxplots
test_evaluation_scores = np.load(project_dir + 'test_evaluation_scores.npy')
plt.boxplot([test_evaluation_scores[:,1,0],test_evaluation_scores[:,2,0],test_evaluation_scores[:,3,0],test_evaluation_scores[:,0,0] ], labels=('AR','LM','SDB','Overall'),sym='x')
plt.ylabel('Precision')
plt.savefig(project_dir + 'precision_boxplots.png', dpi=1200)
plt.show()

# Recall boxplots
test_evaluation_scores = np.load(project_dir + 'test_evaluation_scores.npy')
print(test_evaluation_scores[:,1,2].shape)
plt.boxplot([test_evaluation_scores[:,1,1],test_evaluation_scores[:,2,1],test_evaluation_scores[:,3,1],test_evaluation_scores[:,0,1] ], labels=('AR','LM','SDB','Overall'),sym='x')
plt.ylabel('Recall')
plt.savefig(project_dir + 'recall_boxplots.png', dpi=1200)
plt.show()

# F1-score boxplots
test_evaluation_scores = np.load(project_dir + 'test_evaluation_scores.npy')
plt.boxplot([test_evaluation_scores[:,1,2],test_evaluation_scores[:,2,2],test_evaluation_scores[:,3,2],test_evaluation_scores[:,0,2] ], labels=('AR','LM','SDB','Overall'),sym='x')
plt.ylabel('F1')
plt.savefig(project_dir + 'f1_boxplots.png', dpi=1200)
plt.show()

# ER boxplots
test_evaluation_scores = test_evaluation_scores[test_evaluation_scores[:,2,3]<10]
plt.boxplot([test_evaluation_scores[:,1,3],test_evaluation_scores[:,2,3],test_evaluation_scores[:,3,3],test_evaluation_scores[:,0,3]], labels=('AR','LM','SDB','Overall'),sym='x')
plt.ylabel('ER')
plt.savefig(project_dir + 'er_boxplots.png', dpi=1200)
plt.show()

# test_evaluation_scores = np.load('E:/msc_project/data/test_evaluation_scores.npy')
# test_evaluation_scores = np.mean(test_evaluation_scores[:,:,2], axis=(1))
# highest_f1_files = test_evaluation_scores.argsort()[-24:][::-1]
# print(highest_f1_files)
# 683 787 570 736 197 51 609 302 151 706 289 763  42 838 193

# selected file index
file_id = 838 #570, 838, 476 # Good example file: 838 [12000:17000]
plot_confidence(file_id=file_id)
