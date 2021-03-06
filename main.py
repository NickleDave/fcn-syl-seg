import sys
import os
import shutil
import copy
import logging
import pickle
from datetime import datetime
from configparser import ConfigParser, NoOptionError

import numpy as np
from sklearn.externals import joblib
from keras.utils import to_categorical
import keras.layers
import keras.models
import keras.callbacks

import fcn.models
import fcn.utils

if __name__ == "__main__":
    config_file = sys.argv[1]
    if not config_file.endswith('.ini'):
        raise ValueError('{} is not a valid config file, '
                         'must have .ini extension'
                         .format(config_file))
    config = ConfigParser()
    config.read(config_file)

    timenow = datetime.now().strftime('%y%m%d_%H%M%S')
    if config.has_section('OUTPUT'):
        output_dir = config['OUTPUT']['output_dir']
        results_dirname = os.path.join(output_dir,
                                       'results_' + timenow)
    else:
        results_dirname = os.path.join('.', 'results_' + timenow)
    os.makedirs(results_dirname)
    # copy config file into results dir now that we've made the dir
    shutil.copy(config_file, results_dirname)

    logfile_name = os.path.join(results_dirname,
                                'metadata_from_running_main_'
                                + timenow + '.log')
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
    logger.addHandler(logging.FileHandler(logfile_name))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info('Logging results to {}'.format(results_dirname))
    logger.info('Using config file: {}'.format(config_file))

    spect_params = {}
    for spect_param_name in ['freq_cutoffs', 'thresh']:
        try:
            if spect_param_name == 'freq_cutoffs':
                freq_cutoffs = [float(element)
                                for element in
                                config['SPECTROGRAM']['freq_cutoffs']
                                .split(',')]
                spect_params['freq_cutoffs'] = freq_cutoffs
            elif spect_param_name == 'thresh':
                spect_params['thresh'] = float(config['SPECTROGRAM']['thresh'])
        except NoOptionError:
            logger.info('Parameter for computing spectrogram, '
                        '{}, not specified. '
                        'Will use default.'.format(spect_param_name))
            continue

    print('loading data for training')
    labelset = list(config['DATA']['labelset'])
    label_mapping = dict(zip(labelset,
                             range(1, len(labelset)+1)))
    train_data_dir = config['DATA']['train_data_dir']
    number_train_song_files = int(config['DATA']['number_train_song_files'])
    skip_files_with_labels_not_in_labelset = config.getboolean(
        'DATA',
        'skip_files_with_labels_not_in_labelset')
    train_encoder = config.getboolean('TRAIN', 'train_encoder')
    if train_encoder:
        return_syl_spects = True
        # because we train encoder with spectrograms of individual syllables
        # centered in a window, so that each spectrogram is the same size
        # and so that encoder learns some "context" about neighboring syllables
        try:
            encoder_input_width = float(config['TRAIN']['encoder_input_width'])
        except NoOptionError:
            print('.ini files specifies train_encoder = Yes,'
                  'but no value for encoder_input_width was specified.\n'
                  'Encoder_input_width is required, should be duration'
                  'of spectrogram window centered on syllable in seconds')
            raise  # re-raise NoOptionError
    else:
        return_syl_spects = False
        encoder_input_width = None

    logger.info('Loading training data from {}'.format(train_data_dir))
    logger.info('Using {} song files for training set'
                .format(number_train_song_files))
    return_tup = fcn.utils.load_data(label_mapping,
                                     train_data_dir,
                                     number_train_song_files,
                                     spect_params,
                                     skip_files_with_labels_not_in_labelset,
                                     return_syl_spects,
                                     encoder_input_width)
    # unpack returned tuple
    if return_syl_spects:
        (song_spects, all_labeled_timebin_vectors, masks,
         syl_spects, all_labels,
         timebin_dur, cbins_used) = return_tup
        syl_spects_copy = copy.deepcopy(syl_spects)  # to use for normalizing
        all_labels = np.concatenate(all_labels)
    else:
        (song_spects, all_labeled_timebin_vectors, masks,
         timebin_dur, cbins_used) = return_tup
    cbins_used_filename = os.path.join(results_dirname, 'cbins_used')
    with open(cbins_used_filename, 'wb') as cbins_used_file:
        pickle.dump(cbins_used, cbins_used_file)

    logger.info('Size of each timebin in spectrogram, in seconds: {}'
                .format(timebin_dur))

    # note that training songs are taken from the start of the training data
    # and validation songs are taken starting from the end

    # reshape training data
    num_train_songs = int(config['DATA']['num_train_songs'])
    train_spects = song_spects[:num_train_songs]
    X_train_timebins = np.array(
        [spec.shape[1] for spec in train_spects]
    )  # convert to seconds by multiplying by size of time bin
    X_train_durations = X_train_timebins * timebin_dur
    total_train_set_duration = sum(X_train_durations)
    logger.info('Total duration of training set (in s): {}'
                .format(total_train_set_duration))
    TRAIN_SET_DURS = [int(element)
                      for element in
                      config['TRAIN']['train_set_durs'].split(',')]
    max_train_set_dur = np.max(TRAIN_SET_DURS)

    if max_train_set_dur > total_train_set_duration:
        raise ValueError('Largest duration for a training set of {} '
                         'is greater than total duration of training set, {}'
                         .format(max_train_set_dur, total_train_set_duration))

    logger.info('Will train network with training sets of '
                'following durations (in s): {}'.format(TRAIN_SET_DURS))

    X_train = np.concatenate(train_spects, axis=1)
    # save training set to get training accuracy in summary.py
    joblib.dump(X_train, os.path.join(results_dirname, 'X_train'))
    Y_train = masks[:, :X_train.shape[-1], :]
    assert Y_train.shape[-1] == len(label_mapping) + 1

    num_replicates = int(config['TRAIN']['replicates'])
    REPLICATES = range(num_replicates)
    logger.info('will replicate training {} times '
                'for each duration of training set'
                .format(num_replicates))

    num_val_songs = int(config['DATA']['num_val_songs'])
    logger.info('validation set used during training will contain {} songs'
                .format(num_val_songs))
    if num_train_songs + num_val_songs > number_train_song_files:
        raise ValueError('Total number of training songs ({0}), '
                         'and validation songs ({1}), '
                         'is {2}.\n This is greater than the number of '
                         'songfiles, {3}, and would result in training data '
                         'in the validation set. Please increase the number '
                         'of songfiles or decrease the size of the training '
                         'or validation set.'
                         .format(num_train_songs,
                                 num_val_songs,
                                 num_train_songs + num_val_songs,
                                 number_train_song_files))
    X_val = song_spects[-num_val_songs:]
    joblib.dump(X_val, os.path.join(results_dirname, 'X_val'))
    X_val_copy = copy.deepcopy(X_val)  # need a copy if we scale X_val below
    Y_val = all_labeled_timebin_vectors[-num_val_songs:]
    Y_val_arr = np.concatenate(Y_val, axis=0)

    normalize_spectrograms = config.getboolean('DATA', 'normalize_spectrograms')
    if normalize_spectrograms:
        logger.info('will normalize spectrograms for each training set')

    for train_set_dur in TRAIN_SET_DURS:
        for replicate in REPLICATES:
            logger.info("training with training set duration of {} seconds,"
                        "replicate #{}".format(train_set_dur, replicate))
            training_records_dirname = ('records_for_training_set_'
                                        'with_duration_of_'
                                        + str(train_set_dur) + '_sec_replicate_'
                                        + str(replicate))
            training_records_dirname = os.path.join(results_dirname,
                                                    training_records_dirname)
            checkpoint_filename = ('checkpoint_train_set_dur_'
                                   + str(train_set_dur) +
                                   '_sec_replicate_'
                                   + str(replicate))
            if not os.path.isdir(training_records_dirname):
                os.makedirs(training_records_dirname)
            train_inds = fcn.utils.get_inds_for_dur(X_train_timebins,
                                                    train_set_dur,
                                                    timebin_dur)
            with open(os.path.join(training_records_dirname, 'train_inds'),
                      'wb') as train_inds_file:
                pickle.dump(train_inds, train_inds_file)
            X_train_subset = X_train[:, train_inds]
            Y_train_subset = Y_train[:, train_inds, :]

            if normalize_spectrograms:
                spect_scaler = fcn.utils.SpectScaler()
                X_train_subset = spect_scaler.fit_transform(X_train_subset.T)
                X_train_subset = X_train_subset.T
                if train_encoder:
                    logger.info('normalizing individual syllable spectrograms')
                    syl_spects = np.transpose(syl_spects_copy, axes=[0, 2, 1])
                    syl_spects = spect_scaler.transform(syl_spects)
                    syl_spects = np.transpose(syl_spects_copy, axes=[0, 2, 1])
                logger.info('normalizing validation set to match training set')
                X_val = spect_scaler.transform([x_val_spec.T
                                                for x_val_spec in X_val_copy])
                # rotate back after rotating
                X_val = [x_val_spec.T for x_val_spec in X_val]
                scaler_name = ('spect_scaler_duration_{}_replicate_{}'
                               .format(train_set_dur, replicate))
                joblib.dump(spect_scaler,
                            os.path.join(results_dirname, scaler_name))

            if train_encoder:
                num_syl_spects = int(config['TRAIN']['num_syl_spects'])
                X_train_syl_spects = syl_spects[:num_syl_spects]
                X_train_syl_spects = np.expand_dims(X_train_syl_spects, -1)
                Y_train_syl_spects = all_labels[:num_syl_spects]
                # first, train CNN to classify syllables.
                # Will use those weights for encoder in FCN
                fw = fcn.models.flatwindow(input_shape=X_train_syl_spects.shape[1:],
                                           num_label_classes=len(label_mapping) + 1)
                Y_train_syl_spects = to_categorical(Y_train_syl_spects)
                encoder_epochs = int(config['TRAIN']['encoder_epochs'])
                encoder_batch_size = int(config['TRAIN']['encoder_batch_size'])
                fw.fit(X_train_syl_spects,
                       Y_train_syl_spects,
                       batch_size=encoder_batch_size,
                       epochs=encoder_epochs,
                       validation_split=0.2)
                weights_filename = os.path.join(training_records_dirname,
                                                'encoder_weights.h5')
                fw.save_weights(weights_filename)

            fw_for_fcn = fcn.models.flatwindow(input_shape=X_train_syl_spects.shape[1:],
                                           num_label_classes=len(label_mapping) + 1)
            fcn_width = int(config['TRAIN']['fcn_width'])
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
            Y_train_subset = np.stack(np.split(Y_train_subset, n, axis=1))

            fcn_custom_input_shape = X_train_subset.shape[1:]

            inputs = keras.layers.Input(shape=fcn_custom_input_shape)
            blocks = [fcn.models.fw_conv1(),
                      fcn.models.fw_conv1(),
                      fcn.models.fw_conv2(),
                      fcn.keras_fcn.blocks.vgg_fc(fw_for_fcn.layers[-4].get_config()['units'])]
            encoder = fcn.keras_fcn.encoders.Encoder(inputs,
                                                     blocks,
                                                     weights=weights_filename,
                                                     trainable=True)
            feat_pyramid = encoder.outputs  # A feature pyramid with 5 scales
            feat_pyramid.append(
                inputs)  # Add image to the bottom of the pyramid

            outputs = fcn.keras_fcn.decoders.VGGUpsampler(feat_pyramid,
                                                          scales=[1, 1e-2, 1e-4],
                                                          classes=len(label_mapping) + 1)
            outputs = keras.layers.Activation('softmax')(outputs)

            fcn_custom = keras.models.Model(inputs=inputs, outputs=outputs)

            logs_dir = os.path.join(training_records_dirname,'logs')
            if not os.path.isdir(logs_dir):
                os.makedirs(logs_dir)
            tb = keras.callbacks.TensorBoard(log_dir=logs_dir, histogram_freq=0,
                                        batch_size=32, write_graph=True,
                                        write_grads=False, write_images=False,
                                        embeddings_freq=0, embeddings_layer_names=None,
                                        embeddings_metadata=None)
            fcn_custom.compile(optimizer='rmsprop',
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
            fcn_batch_size = int(config['TRAIN']['fcn_batch_size'])
            fcn_epochs = int(config['TRAIN']['fcn_epochs'])
            fcn_custom.fit(X_train_subset, Y_train_subset,
                           batch_size=fcn_batch_size,
                           epochs=fcn_epochs,
                           callbacks=[tb])
            fcn_model_filename = os.path.join(training_records_dirname,
                                              'fcn_model.h5')
            fcn_custom.save(fcn_model_filename)

