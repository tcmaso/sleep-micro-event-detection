import numpy as np
import os
import xml.etree.ElementTree as ET
import pyedflib
from scipy import signal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa
import librosa.display
from filterbank_shape import FilterbankShape
from variables import *

# Preprocess the data
def preprocess(identifiers, dir_annotations, dir_edfs, dir_save, max_event_lengths, is_test):
    sample_rate = 128
    n_filters = 40
    n_events = 3

    window_length = sample_rate
    if not is_test:
        hop_length = int(sample_rate / 2)
    else:
        hop_length = sample_rate
    # Compute the linear-frequency triangular filter bank
    filter_banks = filtershape.lin_tri_filter_shape(nfilt=n_filters, nfft=window_length, samplerate=32).T

    # Iterate through all files in current set
    for file in identifiers:

        tree = ET.parse(dir_annotations + file + '-nsrr.xml')
        root = tree.getroot()

        n_samples = int(float(root.find("./ScoredEvents/ScoredEvent")[3].text)) * sample_rate
        event_positions = np.zeros((n_events, n_samples))

        # Iterates through all events and assigns a 1 to all samples where the event is active.
        for event in root.findall("./ScoredEvents/ScoredEvent")[1:]:

            event_type = event[0].text.split('|')[0]  # Event Type
            if event_type == 'Arousals':
                j = 0
            elif event_type == 'Limb Movement':
                j = 1
            elif event_type == 'Respiratory':
                j = 2
            else:
                continue

            event_onset = round(float(event[2].text) * sample_rate)  # Event Start

            event_duration = round(float(event[3].text) * sample_rate)  # Event Duration
            if event_duration == 0:
                continue
            elif (event_onset + event_duration) >= n_samples:
                event_duration = n_samples - event_onset -1

            event_offset = event_onset + event_duration  # Event Offset

            for i in range(event_onset, event_offset , 1):
                event_positions[j, i] = 1

        # Read the PSG data for the current file
        edf_reader = pyedflib.EdfReader(dir_edfs + file + '.edf')

        # PSG channels used
        # Arousals: 3,4,7,8,(11,12), LM: 1,2, SDB: 15,16,20
        channels_list = [3,4,7,8,11,1,2,15,16,20]

        spectrograms = []

        #
        for i in channels_list:

            # Resampling of the signals. Chin channels are combined to 1 channel (Right chin substracted from left chin)
            if i == 11:
                channel_signal = signal.resample(edf_reader.readSignal(11), n_samples) - signal.resample(edf_reader.readSignal(12), n_samples)
            else:
                channel_signal = signal.resample(edf_reader.readSignal(i), n_samples)

            # Compute spectrogram and apply linear-frequency triangular filter bank
            spectrogram = librosa.stft(channel_signal, n_fft=window_length, hop_length=hop_length)
            spectrogram = np.dot(filter_banks, spectrogram)
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

            # librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
            # plt.colorbar(format='%+2.0f dB');
            # plt.title('Spectrogram');
            # plt.show()
            # print(spectrogram.shape)

            # Standardizing spectrograms
            means = np.mean(spectrogram, axis=1)
            stds = np.std(spectrogram, axis=1)
            stds[stds==0] = 1

            spectrogram = (spectrogram.transpose() - means) / stds
            spectrogram = spectrogram.transpose()

            spectrograms.append(spectrogram)

        spectrograms = np.array(spectrograms)



        # Number of frames (windows)
        n_frames = spectrograms.shape[2]

        # Localise events to (windows/frames)
        events = np.zeros(shape=(n_frames, 3 * n_events))
        event_onset = 0
        event_offset = 0
        event_duration = 0

        # Compute groundtruth active matrix
        # For the number of frames
        for i in range(n_events):

            for j in range(n_frames):

                # Obtain array of values for current frame
                window = event_positions[i, (j * hop_length):(j * hop_length + window_length)]

                # Intersection over union (IoU)
                IOU = window.sum() / len(window)

                # Frame is assigned 1 (event active) if IOU is greater than/equal to 0.5 and not last frame
                if IOU >= 0.5 and j != n_frames - 1:

                    if event_duration == 0:
                        event_onset = j

                    event_duration += 1

                # Otherwise is assigned 0 (event inactive) if IOU is less than 0.5
                else:

                    if event_duration != 0:

                        event_offset = j - 1

                        # Assign active state to all frames in the event range
                        for k in range(event_onset, event_offset+1, 1):
                            events[k, i*3:(i*3)+3] = [1, (k - event_onset) / (max_event_lengths[i]), (event_offset - k) / (max_event_lengths[i])]

                        event_duration = 0

        # INPUT DATA
        x_data = spectrograms # (n_channels, n_tfilters, n_frames)
        x_data = np.swapaxes(x_data, 0, 2) # (n_frames, n_tfilters, n_channels)

        # OUTPUT DATA
        y_data = events # (n_windows, 3*n_events)

        # Save preprocessed data
        np.save(dir_save + 'x/' + file + '.npy', x_data)
        np.save(dir_save + 'y/'+ file + '.npy', y_data)

        print(x_data.shape)
        print(y_data.shape)
        print(file, 'processed')


# Obtain maximum value of each event class in the training data for normalization
def get_max_event(train_identifiers, dir_annotations, dir_edfs):
    sample_rate = 128
    n_filters = 40
    n_events = 3
    window_length = sample_rate
    hop_length = int(sample_rate / 2)

    max_events = [0,0,0]
    for file in train_identifiers:
        print(file)
        tree = ET.parse(dir_annotations + file + '-nsrr.xml')
        root = tree.getroot()

        n_samples = int(float(root.find("./ScoredEvents/ScoredEvent")[3].text)) * sample_rate

        event_positions = np.zeros((3, n_samples))

        # Iterates through all events and assigns a 1 to all samples where the event is active.
        for event in root.findall("./ScoredEvents/ScoredEvent")[1:]:

            event_type = event[0].text.split('|')[0]  # Event Type
            if event_type == 'Arousals':
                j = 0
            elif event_type == 'Limb Movement':
                j = 1
            elif event_type == 'Respiratory':
                j = 2
            else:
                continue

            event_onset = round(float(event[2].text) * sample_rate)  # Event Start

            event_duration = round(float(event[3].text) * sample_rate)  # Event Duration
            if event_duration == 0:
                continue
            elif (event_onset + event_duration) >= n_samples:
                event_duration = n_samples - event_onset -1

            event_offset = event_onset + event_duration  # Event Offset

            for i in range(event_onset, event_offset , 1):
                event_positions[j, i] = 1

        n_frames = int((n_samples/hop_length)-1)
        n_events = event_positions.shape[0]

        event_duration = 0
        # For the number of windows
        for i in range(n_events):

            for j in range(n_frames):

                # Obtain array of values for current window
                window = event_positions[i, (j * hop_length):(j * hop_length + window_length)]

                # Intersection over union (IoU)
                IOU = window.sum() / len(window)

                # Window is assigned 1 (event active) if IOU is greater than/equal to 0.5
                if IOU >= 0.5 and j != n_frames - 1:

                    event_duration += 1

                # Otherwise is assigned 0 (event inactive) if IOU is less than 0.5
                else:

                    # If current event duration is larger than the current max, then swap with new duration
                    if event_duration != 0 and event_duration > max_events[i]:
                        max_events[i] = event_duration
                    event_duration = 0

    np.save(project_dir + 'training_max_event_lengths', max_events)
    return max_events

# Partition and compute the training, validation and testing sets
def get_identifiers(dir_annotations):
    # Train/Val/Test data indexes
    file_identifiers = []
    for file in os.listdir(dir_annotations):
        file = file.split('-nsrr')[0]
        file_identifiers.append(file)

    # Randomly shuffle identifiers
    train_identifiers, test_identifiers = train_test_split(file_identifiers, test_size=1200, shuffle=True, random_state=10)
    test_identifiers, val_identifiers = train_test_split(test_identifiers, test_size=200, shuffle=False)

    np.save(project_dir + 'train_identifiers.npy', train_identifiers)
    np.save(project_dir + 'val_identifiers.npy', val_identifiers)
    np.save(project_dir + 'test_identifiers.npy', test_identifiers)


# Training, validation, testing identifiers.
if not os.path.isfile(train_identifiers_dir):
    get_identifiers(dir_annotations)
train_identifiers = np.load(train_identifiers_dir)
val_identifiers = np.load(val_identifiers_dir)
test_identifiers = np.load(test_identifiers_dir)


# Max events in training data (for normalization)
if not os.path.isfile(project_dir + 'training_max_event_lengths.npy'):
    max_event_lengths = get_max_event(train_identifiers, dir_annotations, dir_edfs)

max_event_lengths = np.load(project_dir + 'training_max_event_lengths.npy') # [85, 34, 13261]

filtershape = FilterbankShape() # Linear filterbank

# Preprocessing for each set (is_test: True for 50% overlap, False for no overlap)
preprocess(train_identifiers, dir_annotations, dir_edfs, train_dir, max_event_lengths, is_test=False)
preprocess(val_identifiers, dir_annotations, dir_edfs, val_dir, max_event_lengths, is_test=False)
preprocess(test_identifiers, dir_annotations, dir_edfs, test_dir, max_event_lengths, is_test=True)

preprocess(val_identifiers, dir_annotations, dir_edfs, val_inference_dir, max_event_lengths, is_test=True)
