import sys
import os
import pickle
from glob import glob
from configparser import ConfigParser
from datetime import datetime

import numpy as np
from sklearn.externals import joblib
import keras
from keras import backend as K
from keras.layers import Lambda, Activation

import fcn.utils
from fcn import keras_fcn
from keras import Model

config_file = sys.argv[1]
if not config_file.endswith('.ini'):
    raise ValueError('{} is not a valid config file, must have .ini extension'
                     .format(config_file))
config = ConfigParser()
config.read(config_file)

results_dirname = config['OUTPUT']['output_dir']
if not os.path.isdir(results_dirname):
    raise FileNotFoundError('{} directory is not found.'
                            .format(results_dirname))

timenow = datetime.now().strftime('%y%m%d_%H%M%S')
summary_dirname = os.path.join(results_dirname,
                               'summary_' + timenow)
os.makedirs(summary_dirname)

TRAIN_SET_DURS = [int(element)
                  for element in
                  config['TRAIN']['train_set_durs'].split(',')]
num_replicates = int(config['TRAIN']['replicates'])
REPLICATES = range(num_replicates)
normalize_spectrograms = config.getboolean('DATA', 'normalize_spectrograms')

spect_params = {}
for spect_param_name in ['freq_cutoffs', 'thresh']:
    try:
        if spect_param_name == 'freq_cutoffs':
            freq_cutoffs = [float(element)
                            for element in
                            config['SPECTROGRAM']['freq_cutoffs'].split(',')]
            spect_params['freq_cutoffs'] = freq_cutoffs
        elif spect_param_name == 'thresh':
            spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])

    except NoOptionError:
        logger.info('Parameter for computing spectrogram, {}, not specified. '
                    'Will use default.'.format(spect_param_name))
        continue
if spect_params == {}:
    spect_params = None

print('loading training data')
labelset = list(config['DATA']['labelset'])
label_mapping = dict(zip(labelset,
                         range(1, len(labelset) + 1)))
train_data_dir = config['DATA']['train_data_dir']
number_train_song_files = int(config['DATA']['number_train_song_files'])
skip_files_with_labels_not_in_labelset = config.getboolean(
    'DATA',
    'skip_files_with_labels_not_in_labelset')
# don't need syl spects for measuring accuracy
return_syl_spects = False
encoder_input_width = None

return_tup = fcn.utils.load_data(label_mapping,
                                 train_data_dir,
                                 number_train_song_files,
                                 spect_params,
                                 skip_files_with_labels_not_in_labelset,
                                 return_syl_spects,
                                 encoder_input_width)

(song_spects, all_labeled_timebin_vectors, masks,
 timebin_dur, cbins_used) = return_tup

# reshape training data
num_train_songs = int(config['DATA']['num_train_songs'])
train_spects = song_spects[:num_train_songs]
X_train = np.concatenate(train_spects, axis=1)
Y_train = np.concatenate(all_labeled_timebin_vectors).ravel()

print('loading testing data')
test_data_dir = config['DATA']['test_data_dir']
number_test_song_files = int(config['DATA']['number_test_song_files'])
return_tup = fcn.utils.load_data(label_mapping,
                                 test_data_dir,
                                 number_test_song_files,
                                 spect_params,
                                 skip_files_with_labels_not_in_labelset,
                                 return_syl_spects,
                                 encoder_input_width)

(test_song_spects, test_labeled_timebin_vectors, test_masks,
 timebin_dur, cbins_used) = return_tup
cbins_used_filename = os.path.join(results_dirname, 'test_cbins_used')
with open(cbins_used_filename, 'wb') as cbins_used_file:
    pickle.dump(cbins_used, cbins_used_file)

# X_test_copy because X_test gets scaled and reshaped in main loop
X_test_copy = np.concatenate(test_song_spects, axis=1)
# make Y_test here because it doesn't change
Y_test = np.concatenate(test_labeled_timebin_vectors).ravel()

fcn_width = int(config['TRAIN']['fcn_width'])
inds_to_split_Y_test = np.arange(start=fcn_width,
                                 stop=test_masks.shape[1],
                                 step=fcn_width)
Y_test = np.stack(np.split(Y_test, inds_to_split_Y_test)[:-1])

normalize_spectrograms = config.getboolean('DATA', 'normalize_spectrograms')

TRAIN_SET_DURS = [int(element)
                  for element in
                  config['TRAIN']['train_set_durs'].split(',')]
num_replicates = int(config['TRAIN']['replicates'])
REPLICATES = range(num_replicates)

# initialize arrays to hold summary results
train_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
test_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
Y_pred_test_all = []  # will be a nested list
Y_pred_train_all = [] # will be a nested list


for dur_ind, train_set_dur in enumerate(TRAIN_SET_DURS):

    Y_pred_test_this_dur = []
    Y_pred_train_this_dur = []

    for rep_ind, replicate in enumerate(REPLICATES):
        print("getting train and test error for "
              "training set with duration of {} seconds, "
              "replicate {}".format(train_set_dur, replicate))
        training_records_dir = os.path.join(results_dirname,
                                            ('records_for_training_set_with_duration_of_'
                                             + str(train_set_dur) + '_sec_replicate_'
                                             + str(replicate))
                                            )
        train_inds_file = glob(os.path.join(training_records_dir, 'train_inds'))[0]
        with open(os.path.join(train_inds_file), 'rb') as train_inds_file:
            train_inds = pickle.load(train_inds_file)

        # get training set
        X_train_subset = X_train[:, train_inds]
        Y_train_subset = Y_train[train_inds]
        # normalize before reshaping to avoid even more convoluted array reshaping
        if normalize_spectrograms:
            print('normalizing spectrograms')
            scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                           .format(train_set_dur, replicate))
            spect_scaler = joblib.load(os.path.join(results_dirname, scaler_name))
            X_train_subset = spect_scaler.transform(X_train_subset.T).T
            X_test = spect_scaler.transform(X_test_copy.T).T

        # now that we normalized, we can reshape
        # note we loaded fcn_width from .ini file above
        if X_train_subset.shape[-1] % fcn_width != 0:
            raise ValueError('Duration of X_train_subset, {}, '
                             'is not evenly divisible into segments of'
                             'width specified for FCN, {}.\nWould result in loss of'
                             'training data.'
                             .format(X_train_subset.shape[-1], fcn_width))
        n = np.arange(start=fcn_width,
                      stop=X_train_subset.shape[-1],
                      step=fcn_width)
        X_train_subset = np.stack(np.split(X_train_subset, n, axis=1))
        X_train_subset = X_train_subset[:, :, :, np.newaxis]
        Y_train_subset = np.stack(np.split(Y_train_subset, n))
        # below don't keep last array because it might be less wide than fcn_width
        # since test set is not guaranteed to be of length
        # evenly divisible by fcn_width
        inds_to_split_X_test = np.arange(start=fcn_width,
                                 stop=X_test_copy.shape[-1],
                                 step=fcn_width)
        # sanity check
        assert np.array_equal(inds_to_split_X_test, inds_to_split_Y_test)
        X_test = np.stack(np.split(X_test, inds_to_split_X_test, axis=1)[:-1])
        X_test = X_test[:, :, :, np.newaxis]

        model_filename = os.path.join(training_records_dir,
                                      'fcn_model.h5')
        fcn = keras.models.load_model(model_filename,
                                      custom_objects={'BilinearUpSampling2D':
                                                          keras_fcn.layers.BilinearUpSampling2D})
        outputs2 = Lambda(lambda x: K.sum(x, axis=1))(fcn.output)
        outputs2 = Activation('softmax')(outputs2)
        fcn_custom2 = Model(inputs=fcn.input, outputs=outputs2)
        # optimizer='SGD' is not conservative enough, it can fail during training
        fcn_custom2.compile(optimizer='rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
        Y_pred_train = fcn_custom2.predict(X_train_subset)
        Y_pred_test = fcn_custom2.predict(X_test)

        Y_pred_train_max = np.argmax(Y_pred_train, axis=2)
        Y_pred_test_max = np.argmax(Y_pred_test, axis=2)

        train_err = np.sum(Y_pred_train_max.ravel()
                           - Y_train_subset.ravel() != 0) / Y_train_subset.ravel().shape[-1]
        train_err_arr[dur_ind, rep_ind] = train_err
        print('train error was {}'.format(train_err))

        test_err = np.sum(Y_pred_test_max.ravel()
                          - Y_test.ravel() != 0) / Y_test.ravel().shape[-1]
        test_err_arr[dur_ind, rep_ind] = test_err
        print('test error was {}'.format(test_err))

        Y_pred_train_this_dur.append(Y_pred_train)
        Y_pred_test_this_dur.append(Y_pred_test)

    Y_pred_train_all.append(Y_pred_train_this_dur)
    Y_pred_test_all.append(Y_pred_test_this_dur)

Y_pred_train_filename = os.path.join(training_records_dir,
                                  'Y_pred_train')
with open(Y_pred_train_filename,'wb') as Y_pred_train_file:
    pickle.dump(Y_pred_train_all, Y_pred_train_file)

Y_pred_test_filename = os.path.join(training_records_dir,
                                  'Y_pred_test')
with open(Y_pred_test_filename,'wb') as Y_pred_test_file:
    pickle.dump(Y_pred_test_all, Y_pred_test_file)

train_err_filename = os.path.join(training_records_dir,
                                  'train_err')
with open(train_err_filename,'wb') as train_err_file:
    pickle.dump(train_err_arr, train_err_file)

test_err_filename = os.path.join(training_records_dir,
                                  'test_err')
with open(test_err_filename, 'wb') as test_err_file:
    pickle.dump(test_err_arr, test_err_file)
