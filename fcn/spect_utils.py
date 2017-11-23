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

    all_syls = []

    for ind, (label, onset, offset) in enumerate(zip(labels, onsets, offsets)):

        syl_duration = offset - onset
        if syl_duration > syl_spect_width:
            raise ValueError('syllable duration of syllable {} with label {} '
                             'in file {} is greater than '
                             'width specified for all syllable spectrograms.'
                             .format(ind, label, filename))

        width_diff = syl_spect_width_Hz - syl_duration_in_samples
        # take half of difference between syllable duration and spect width
        # so one half of 'empty' area will be on one side of spect
        # and the other half will be on other side
        # i.e., center the spectrogram
        left_width = int(round(width_diff / 2))
        right_width = width_diff - left_width
        if left_width > onset:  # if duration before onset is less than left_width
            # (could happen with first onset)
            syl_audio = self.rawAudio[0:syl_spect_width_Hz]
        elif offset + right_width > self.rawAudio.shape[-1]:
            # if right width greater than length of file
            syl_audio = self.rawAudio[-syl_spect_width_Hz:]
        else:
            syl_audio = self.rawAudio[onset - left_width:offset + right_width]

        except WindowError as err:
            warnings.warn('Segment {0} in {1} with label {2} '
                          'not long enough for window function'
                          ' set with current spect_params.\n'
                          'spect will be set to nan.'
                          .format(ind, self.filename, label))
            spect, freq_bins, time_bins = (np.nan,
                                           np.nan,
                                           np.nan)


        all_syls.append(curr_syl)

    return np.stack([syl.spect for syl in all_syls])