import os
from glob import glob
import random

import numpy as np

from . import evfuncs
from .spect_utils import butter_bandpass_filter, spectrogram, make_syl_spects


# adapted from:
# https://github.com/NickleDave/hybrid-vocal-classifier/blob/master/hvc/neuralnet/utils.py
class SpectScaler:
    """class that scales spectrograms that all have the
    same number of frequency bins. Any input spectrogram
    will be scaled by subtracting off the mean of each
    frequency bin from the 'fit' set of spectrograms, and
    then dividing by the standard deviation of each
    frequency bin from the 'fit' set.
    """

    def __init__(self):
        pass

    def fit(self, spect):
        """fit a SpectScaler.
        Input should be spectrogram,
        oriented so that the columns are frequency bins.
        Fit function finds the mean and standard deviation of
        each frequency bin, which are used by `transform` method
        to scale other spectrograms.

        Parameters
        ----------
        spect : 2-d numpy array
            with dimensions (time bins, frequency bins)
        """

        if spect.ndim != 2:
            raise ValueError('input spectrogram should be a 2-d array')

        self.columnMeans = np.mean(spect, axis=0)
        self.columnStds = np.std(spect, axis=0)
        assert self.columnMeans.shape[-1] == spect.shape[-1]
        assert self.columnStds.shape[-1] == spect.shape[-1]
        self.nonZeroStd = np.argwhere(self.columnStds != 0)

    def _transform(self, spect):
        """transforms input spectrogram by subtracting off fit mean
        and then dividing by standard deviation
        """

        if spect.ndim == 2:
            transformed = spect - self.columnMeans
            # to keep any zero stds from causing NaNs
            transformed[:, self.nonZeroStd] = (
                transformed[:, self.nonZeroStd] / self.columnStds[self.nonZeroStd])
        elif spect.ndim == 3:
            transformed = np.copy(spect)
            for ind in np.arange(spect.shape[0]):
                transformed[ind] = spect[ind] - self.columnMeans
                # to keep any zero stds from causing NaNs
                transformed[ind][:, self.nonZeroStd] = (
                    transformed[ind][:, self.nonZeroStd] / self.columnStds[
                        self.nonZeroStd])
        return transformed

    def transform(self, spects):
        """normalizes input spectrograms with fit parameters
        Assumes spectrograms are oriented with columns being frequency bins
        and rows being time bins.

        Parameters
        ----------
        spects : 2-d numpy array or list of 2-d numpy arrays
            with dimensions (time bins, frequency bins)

        """

        if any([not hasattr(self, attr) for attr in ['columnMeans',
                                                     'columnStds']]):
            raise AttributeError('SpectScaler properties are set to None,'
                                 'must call fit method first to set the'
                                 'value of these properties before calling'
                                 'transform')

        if type(spects) != np.ndarray and type(spects) != list:
            raise TypeError('type {} is not valid for spects'
                            .format(type(spects)))

        if type(spects) == np.ndarray:
            if spects.shape[-1] != self.columnMeans.shape[-1]:
                raise ValueError('number of columns in spects, {}, '
                                 'does not match shape of self.columnMeans, {},'
                                 'i.e. the number of columns from the spectrogram'
                                 'to which the scaler was fit originally')
            return self._transform(spects)


        elif type(spects) == list:
            z_norm_spects = []
            for spect in spects:
                z_norm_spects.append(self._transform(spect))

            return z_norm_spects

    def fit_transform(self, spects):
        """first calls fit and then returns normalized spects
        transformed using the fit parameters"""

        if type(spects) != np.ndarray:
            raise TypeError('spects passed to fit_transform '
                            'should be numpy array, not {}'
                            .format(type(spects)))

        if spects.ndim != 2:
            raise ValueError('ndims of spects should be 2, not {}'
                             .format(spects.ndim))

        self.fit(spects)
        return self.transform(spects)


def make_labels_mapping(data_dirs):
    """make_labels_mapping_mapping
    from the set of unique string labels: [i,a,b,c,d,h,j,k]
    to a sequence of integers: [0,1,2,3,...
    for converting labels into integers
    that can then be converted to one-hot vectors
    for training outputs of a neural network"""

    if type(data_dirs) == str:  # if just one directory name
        # put into a list so code below works
        data_dirs = [data_dirs]

    if type(data_dirs) != list:
        raise TypeError('data_dirs should be a list but is {}'
                        .format(type(data_dirs)))

    labels = []
    for data_dir in data_dirs:
        notmats = glob(data_dir + '*.not.mat')
        for notmat in notmats:
            notmat_dict = evfuncs.load_notmat(notmat)
            labels.extend(list(notmat_dict['labels']))
    uniq_labels = np.unique(labels)
    labels_to_map_to = range(1, uniq_labels.shape[-1] + 1)
    # skip 0 so 0 can be used as label for 'silent gap' across training/testing data
    return dict(zip(uniq_labels, labels_to_map_to))


def make_labeled_timebins_vector(labels,
                                 onsets,
                                 offsets,
                                 time_bins,
                                 silent_gap_label=0):
    """makes a vector of labels for each timebin from a spectrogram,
    given labels for syllables plus onsets and offsets of syllables

    Parameters
    ----------
    labels : ints
        should be mapping returned by make_labels_mapping
    onsets : ndarray
        1d vector of floats, syllable onsets in seconds
    offsets : ndarray
        1d vector of floats, offsets in seconds
    time_bins : ndarray
        1d vector of floats,
        time in seconds for each time bin of a spectrogram
    silent_gap_label : int
        label assigned to silent gaps
        default is 0

    Returns
    -------

    """

    labels = [int(label) for label in labels]
    label_vec = np.ones((time_bins.shape[-1], 1), dtype='int8') * silent_gap_label
    onset_inds = [np.argmin(np.abs(time_bins - onset))
                  for onset in onsets]
    offset_inds = [np.argmin(np.abs(time_bins - offset))
                   for offset in offsets]
    for label, onset, offset in zip(labels, onset_inds, offset_inds):
        label_vec[onset:offset+1] = label
        # offset_inds[ind]+1 because of Matlab one-indexing
    return label_vec


def load_data(labels_mapping,
              data_dir,
              number_files,
              spect_params,
              skip_files_with_labels_not_in_labelset,
              return_syl_spects=False,
              syl_spect_width=None):
    """

    Parameters
    ----------
    labels_mapping : dict
        dict that maps string labels to a integer values 0 through n,
        where n is the number of labels
    data_dir : str
        directory of data
    number_files : int
        number of files in list of song files to process
        assumes files is cbins
    spect_params : dict
        parameters for computing spectrogram. Loaded by main.py from .ini file.
    skip_files_with_labels_not_in_labelset : bool
        if True, skips files where labels includes a label not found in labels_mapping.
        To avoid training/testing on labels that rarely occur.
        Default is True.
    return_syl_spects : bool
        if True, return spectrograms for each syllable
    syl_spect_width : float
        Parameter to set constant duration for each spectrogram of a
        syllable, in seconds.
        E.g., 0.3 for 300 ms window centered on syllable.
        Default is 0.12, 120 ms.

    Returns
    -------
    spects : list
        of 2-d ndarrays, spectrograms
    labels : list
        of strings, labels corresponding to each spectrogram
    timebin_dur : float
        duration of a timebin in seconds from spectrograms
        estimated from last spectrogram processed
    """

    if return_syl_spects and syl_spect_width is None:
        raise ValueError('return_syl_spects set to True but '
                         'no value specified for syl_spect_width')

    all_song_spects = []
    all_labels = []
    all_labeled_timebin_vecs = []
    counter = 0
    # need to keep track of name of files used
    # since we may skip some
    cbins_used = []
    if not os.path.isdir(data_dir):
        raise NotADirectoryError('String supllied for data_dir, '
                                 '{}, is not a valid directory'
                                 .format(data_dir))
    cbin_str = os.path.join(data_dir, '*.cbin')
    cbins = glob(cbin_str)
    for cbin in cbins:
        dat, fs = evfuncs.load_cbin(cbin)
        notmat_dict = evfuncs.load_notmat(cbin)
        labels = notmat_dict['labels']
        if skip_files_with_labels_not_in_labelset:
            labels_set = set(labels)
            # below, set(labels_mapping) is a set of that dict's keys
            if labels_set > set(labels_mapping):
                # because there's some label in labels
                # that's not in labels_mapping
                continue  # skip that file

        if 'freq_cutoffs' in spect_params:
            dat = butter_bandpass_filter(dat,
                                         spect_params['freq_cutoffs'][0],
                                         spect_params['freq_cutoffs'][1],
                                         fs)

        spect, freqbins, timebins = spectrogram(dat,
                                                fs,
                                                thresh=spect_params['thresh'])
        if 'freq_cutoffs' in spect_params:
            # f_inds = np.nonzero(...)[0] because nonzero returns tuple
            f_inds = np.nonzero((freqbins >= spect_params['freq_cutoffs'][0]) &
                                (freqbins < spect_params['freq_cutoffs'][1]))[0]
            spect = spect[f_inds, :]

        all_song_spects.append(spect)

        labels = [labels_mapping[label]
                  for label in notmat_dict['labels']]
        # get onsets and offsets, divide by 1000 to convert from ms to s
        onsets = notmat_dict['onsets']/1000
        offsets = notmat_dict['offsets']/1000
        labeled_timebin_vec = make_labeled_timebins_vector(labels,
                                                           onsets,
                                                           offsets,
                                                           timebins)
        all_labels.append(labels)
        all_labeled_timebin_vecs.append(labeled_timebin_vec)

        if return_syl_spects:
            syl_spects = make_syl_spects(spect,
                                         cbin,
                                         timebins,
                                         labels,
                                         onsets,
                                         offsets,
                                         syl_spect_width)
            if 'all_syl_spects' in locals():
                all_syl_spects = np.concatenate((all_syl_spects, syl_spects),
                                                axis=0)
            else:
                all_syl_spects = syl_spects

        counter = counter + 1
        cbins_used.append(cbin)
        if counter == number_files:
            break

    if counter < number_files:
        raise ValueError('Less than number of files requested, {}, '
                         'in directory {}'
                         .format(number_files, data_dir))
    timebin_dur = np.around(np.mean(np.diff(timebins)), decimals=3)

    return_tup = (all_song_spects, all_labeled_timebin_vecs)
    if return_syl_spects:
        return_tup += (all_syl_spects, all_labels)
    return_tup += (timebin_dur, cbins_used)
    return return_tup


def get_inds_for_dur(song_timebins,
                     target_duration,
                     timebin_dur_in_s=0.001):
    """for getting a training set with random songs but constant duration
    draws songs at random and adds to list
    until total duration of all songs => target_duration
    then truncates at target duration

    Parameters
    ----------
    song_timebins : list
        list of number of timebins for each songfile,
        where timebines is number of rows in a spectrogram
        e.g.,
        [song_spect.shape[0] for song_spect in song_spectrograms]
        (rows are time instead of frequency,
        because network is set up with input this way)
    target_duration : float
        target duration of training set in s
    timebin_dur_in_s : float
        duration of each timebin, i.e. each column in spectrogram,
        in seconds.
        default is 0.001 s (1 ms)

    Returns
    -------
    inds_to_use : bool
        numpy boolean vector, True where row in X_train gets used
        (assumes X_train is one long spectrogram, consisting of all
        training spectrograms concatenated, and each row being one timebin)
    """

    for song_ind, num_timebins_in_song in enumerate(song_timebins):
        inds = np.ones((num_timebins_in_song,), dtype=int) * song_ind
        if 'song_inds_arr' in locals():
            song_inds_arr = np.concatenate((song_inds_arr, inds))
        else:
            song_inds_arr = inds

    song_id_list = []
    total_dur_in_timebins = 0
    num_songs = len(song_timebins)
    while 1:
        song_id = random.randrange(num_songs)
        if song_id in song_id_list:
            continue
        else:
            song_id_list.append(song_id)
            song_id_inds = np.where(song_inds_arr == song_id)[0]  # 0 because where np.returns tuple
            if 'inds_to_use' in locals():
                inds_to_use = np.concatenate((inds_to_use, song_id_inds))
            else:
                inds_to_use = song_id_inds
            total_dur_in_timebins = total_dur_in_timebins + song_timebins[song_id]
            if total_dur_in_timebins * timebin_dur_in_s >= target_duration:
                # if total_dur greater than target, need to truncate
                if total_dur_in_timebins * timebin_dur_in_s > target_duration:
                    correct_length = np.round(target_duration / timebin_dur_in_s).astype(int)
                    inds_to_use = inds_to_use[:correct_length]
                # (if equal to target, don't need to do anything)
                break

    return inds_to_use


def reshape_data_for_batching(X, Y, batch_size, time_steps, input_vec_size):
    """reshape to feed to network in batches"""
    # need to loop through train data in chunks, can't fit on GPU all at once
    # First zero pad
    num_batches = X.shape[0] // batch_size // time_steps
    rows_to_append = ((num_batches + 1) * time_steps * batch_size) - X.shape[0]
    X = np.concatenate((X, np.zeros((rows_to_append, input_vec_size))),
                       axis=0)
    Y = np.concatenate((Y, np.zeros((rows_to_append, 1), dtype=int)), axis=0)
    num_batches = num_batches + 1
    X = X.reshape((batch_size, num_batches * time_steps, -1))
    Y = Y.reshape((batch_size, -1))
    return X, Y, num_batches