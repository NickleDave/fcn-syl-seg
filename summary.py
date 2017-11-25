import sys
import os
import pickle
from glob import glob
from configparser import ConfigParser

import numpy as np
from sklearn.externals import joblib
import keras
from keras import backend as K
from keras.layers import Lambda, Activation

import fcn.utils

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

train_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))
test_err_arr = np.empty((len(TRAIN_SET_DURS), len(REPLICATES)))

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
Y_train = masks[:, :X_train.shape[-1], :]
assert Y_train.shape[-1] == len(label_mapping) + 1

print('loading testing data')
test_data_dir = config['DATA']['test_data_dir']
number_test_song_files = int(config['DATA']['number_test_song_files'])
return_tup = fcn.utils.load_data(label_mapping,
                                 train_data_dir,
                                 number_train_song_files,
                                 spect_params,
                                 skip_files_with_labels_not_in_labelset,
                                 return_syl_spects,
                                 encoder_input_width)

(test_song_spects, test_labeled_timebin_vectors, test_masks,
 timebin_dur, cbins_used) = return_tup
cbins_used_filename = os.path.join(results_dirname, 'test_cbins_used')
with open(cbins_used_filename, 'wb') as cbins_used_file:
    pickle.dump(cbins_used, cbins_used_file)

X_test = np.concatenate(test_song_spects, axis=1)
# copy X_test because it gets scaled and reshape in main loop
X_test_copy = np.copy(X_test)
Y_test = np.concatenate(test_masks, axis=1)
# also need copy of Y_test
# because it also gets reshaped in loop
# and because we need to compare with Y_pred
Y_test_copy = np.copy(Y_test)

normalize_spectrograms = config.getboolean('DATA', 'normalize_spectrograms')

TRAIN_SET_DURS = [int(element)
                  for element in
                  config['TRAIN']['train_set_durs'].split(',')]
num_replicates = int(config['TRAIN']['replicates'])
REPLICATES = range(num_replicates)


for dur_ind, train_set_dur in enumerate(TRAIN_SET_DURS):
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
        Y_train_subset = Y_train[:, train_inds, :]
        # normalize before reshaping to avoid even more convoluted array reshaping
        if normalize_spectrograms:
            scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                           .format(train_set_dur, replicate))
            spect_scaler = joblib.load(os.path.join(results_dirname, scaler_name))
            X_train_subset = spect_scaler.transform(X_train_subset.T).T
            X_test = spect_scaler.transform(X_test_copy.T).T
            Y_test = np.copy(Y_test_copy)

        # now that we normalized, we can reshape
        fcn_width = int(config['TRAIN']['fcn_width'])
        n = np.arange(start=fcn_width,
                      stop=X_train_subset.shape[-1],
                      step=fcn_width)
        X_train_subset = np.stack(np.split(X_train_subset, n, axis=1))
        X_train_subset = X_train_subset[:, :, :, np.newaxis]
        Y_train_subset = np.stack(np.split(Y_train_subset, n, axis=1))
        # below don't keep last array because it might be less wide than fcn_width
        X_test = np.stack(np.split(X_test, n, axis=1)[-1])
        X_test = X_test[:, :, :, np.newaxis]
        Y_test = np.stack(np.split(Y_test, n, axis=1))


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

        y_pred_train = []
        for ind in range(X_train_subset.shape[0]):
            y_pred.append(fcn_custom2.predict(spects_split[ind:ind + 1, :, :, :]))

        import pdb;pdb.set_trace()

train_err_filename = os.path.join(training_records_dir,
                                  'train_err')
with open(train_err_filename,'wb') as train_err_file:
    pickle.dump(train_err_arr, train_err_file)

test_err_filename = os.path.join(training_records_dir,
                                  'test_err')
with open(test_err_filename, 'wb') as test_err_file:
    pickle.dump(test_err_arr, test_err_file)
