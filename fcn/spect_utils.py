"""spectrogram utilities
filters adapted from SciPy cookbook
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
spectrogram adapted from code by Kyle Kastner and Tim Sainburg
https://github.com/timsainb/python_spectrograms_and_inversion
"""

import numpy as np
from scipy.signal import butter, lfilter
from matplotlib.mlab import specgram


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def spectrogram(data, samp_freq, fft_size=512, step_size=64, thresh=6.25, log=True):
    """creates a spectrogram
    data : ndarray
        audio signal
    log: bool
        if True, take the log of the spectrogram
    thresh: int
        threshold minimum power for log spectrogram
    """

    noverlap = fft_size - step_size

    # below only take [:3] from return of specgram because we don't need the image
    spec, freqbins, timebins = specgram(data, fft_size, samp_freq, noverlap=noverlap)[:3]

    if log:
        spec /= spec.max()  # volume normalize to max 1
        spec = np.log10(spec)  # take log
        spec[spec < -thresh] = -thresh  # set anything less than the threshold as the threshold
    else:
        spec[spec < thresh] = thresh  # set anything less than the threshold as the threshold

    return spec, freqbins, timebins


def make_syl_spects(spect,
                    filename,
                    timebins,
                    labels,
                    onsets,
                    offsets,
                    syl_spect_width=0.12):
    """Extract spectrogram for each syllable in a song.

    Parameters
    ----------
    spect : ndarray
        2d array containing spectrogram of entire song
    filename : str
        filename from which spectrogram was taken
    labels : str
        labels for syllables in song
    onsets : ndarray
        1d array of onset times in seconds
    offsets : ndarray
        1d array of offset times in seconds
    syl_spect_width : float
        Parameter to set constant duration for each spectrogram of a
        syllable, in seconds.
        E.g., 0.3 for 300 ms window centered on syllable.
        Default is 0.12, 120 ms.
    """

    if syl_spect_width > 1:
        warnings.warn('syl_spect_width set greater than 1; note that '
                      'this parameter is in units of seconds, so using '
                      'a value greater than one will make it hard to '
                      'center the syllable/segment of interest within'
                      'the spectrogram, and additionally consume a lot '
                      'of memory.')

    timebin_dur = np.around(np.mean(np.diff(timebins)), decimals=3)
    syl_spect_width_bins = syl_spect_width / timebin_dur
    if not syl_spect_width_bins.is_integer():
        raise ValueError('the syl_spect_width in seconds, {},'
                         'divided by the duration of each time bin,'
                         '{}, does not give a whole number of bins.'
                         'Instead it gives {}.'
                         .format(syl_spect_width,
                                 timebin_dur,
                                 syl_spect_width_bins))
    else:
        # need to convert to int (type) to use for indexing
        syl_spect_width_bins = int(syl_spect_width_bins)

    all_syl_spects = np.empty((len(labels),
                               spect.shape[0],
                               syl_spect_width_bins))
    onset_IDs_in_time_bins = [np.argmin(np.abs(timebins - onset)) for onset in
                              onsets]
    offset_IDs_in_time_bins = [np.argmin(np.abs(timebins - offset)) for offset
                               in offsets]

    for ind, (label, onset, offset) in enumerate(zip(labels,
                                                     onset_IDs_in_time_bins,
                                                     offset_IDs_in_time_bins)):
        syl_duration = offset - onset
        if syl_duration > syl_spect_width_bins:
            raise ValueError('syllable duration of syllable {} with label {} '
                             'in file {} is greater than '
                             'width specified for all syllable spectrograms.'
                             .format(ind, label, filename))
        width_diff = syl_spect_width_bins - syl_duration
        # take half of difference between syllable duration and spect width
        # so one half of 'empty' area will be on one side of spect
        # and the other half will be on other side
        # i.e., center the spectrogram
        left_width = int(round(width_diff / 2))
        right_width = width_diff - left_width
        if left_width > onset:
            # if duration before onset is less than left_width
            # (could happen with first onset)
            # then just align start of window with start of song
            syl_spect = spect[:, 0:syl_spect_width_bins]
        elif offset + right_width > spect.shape[-1]:
            # if right width greater than length of file
            # then just align end of window at end of song
            syl_spect = spect[:, -syl_spect_width_bins:]
        else:
            syl_spect = spect[:, onset - left_width:offset + right_width]

        all_syl_spects[ind, :, :] = syl_spect

    return all_syl_spects
