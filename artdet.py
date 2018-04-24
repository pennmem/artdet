"""Rejection of artifactual and saturated channels."""

import random

import numpy as np
import pandas as pd
from traits.api import HasTraits, Array, CArray, Int, ListStr

# dtype for events recarrays
_events_dtype = [
    ('channel', '|U32'),
    ('frequency', np.int),  # mHz
    ('amplitude', np.int),  # uA
    ('duration', np.int),  # ms
    ('onset', np.int),  # ms
    ('hashtag', '|S40'),  # sha1 hash
]


class ArtifactDetector(HasTraits):
    """Class used to detect artifactual and saturated channels."""
    timeseries = Array(desc='time series data, shape = (n_channels, n_samples)')
    channels = ListStr(desc='channel labels')
    stim_events = Array(desc='stimulation events', dtype=_events_dtype)
    sham_events = Array(desc='sham stimulation events', dtype=_events_dtype)
    pre_stim_start = Int(-440, desc='pre-stim start time relative to stim')
    pre_stim_stop = Int(-40, desc='pre-stim stop time relative to stim')
    post_stim_start = Int(40, desc='post-stim start time relative to stim')
    post_stim_stop = Int(440, desc='post-stim stop time relative to stim')

    def __init__(self, timeseries, channels, stim_events, sham_events):
        super(ArtifactDetector, self).__init__()

        self.timeseries = timeseries
        self.channels = channels
        self.stim_events = stim_events
        self.sham_events = sham_events

        # defer to when we get bad channels because we may wish to tweak start
        # and stop times
        self._pre_stim_intervals = None  # type: np.ndarray
        self._post_stim_intervals = None  # type: np.ndarray
        self._pre_sham_intervals = None  # type: np.ndarray
        self._post_sham_intervals = None  # type: np.ndarray

    def _extract_intervals(self, sham=False):
        """Extract pre- and post-stim/sham intervals.

        Parameters
        ----------
        sham : bool
            Whether to extract sham (True) or stim (False) intervals.

        Returns
        -------
        dict
            A dict containing pre and post intervals which are arrays with
            dimensions (n_events, n_channels, n_samples_per_event)

        """
        events = self.sham_events if sham else self.stim_events

        diff = lambda a, b: max(a, b) - min(a, b)

        pre_length = diff(self.pre_stim_start, self.pre_stim_stop)
        post_length = diff(self.post_stim_start, self.post_stim_stop)

        pre = np.zeros((len(events), len(self.channels), pre_length))
        post = np.zeros((len(events), len(self.channels), post_length))

        for i, onset in enumerate(events.onset):
            pre[i, :] = self.timeseries[:, (onset + self.pre_stim_start):(onset + self.pre_stim_stop)]
            post[i, :] = self.timeseries[:, (onset + self.post_stim_start):(onset + self.post_stim_stop)]

        return {
            'pre': pre,
            'post': post,
        }

    def _get_saturated_channels(self):
        """Identify channels which display post-stim saturation.

        Returns
        -------
        rejected : dict
            A dictionary with channel labels as keys and boolean
            values (True meaning the channel is rejected).

        Notes
        -----
        Stimulation channels are not treated any differently by
        this method.

        """
        return {key: random.choice([True, False]) for key in self.channels}

    def _get_artifactual_channels(self):
        """Identify channels which display significant post-stim artifact.

        Returns
        -------
        rejected : dict
            See identify_saturated_channels

        Notes
        -----
        Stimulation channels are not treated any differently by
        this method.

        """
        return {key: random.choice([True, False]) for key in self.channels}

    def get_bad_channels(self):
        """Identify all bad channels.

        Returns
        -------
        bad_channels : pd.DataFrame
            DataFrame identifying channels as rejected or not (along with reason
            for rejection)

        """
        stim_intervals = self._extract_intervals(False)
        self._pre_stim_intervals = stim_intervals['pre']
        self._post_stim_intervals = stim_intervals['post']

        sham_intervals = self._extract_intervals(True)
        self._pre_sham_intervals = sham_intervals['pre']
        self._post_sham_intervals = sham_intervals['post']

        saturated = self._get_saturated_channels()
        artifactual = self._get_artifactual_channels()

        rejected = [
            (channel, saturated[channel], artifactual[channel])
            for channel in self.channels
        ]

        bad_channels = pd.DataFrame(rejected, columns=('channel', 'saturated', 'artifactual'))
        return bad_channels


if __name__ == "__main__":
    from hashlib import sha1

    def hashit(x):
        return sha1(str(x).encode())

    channels = ['LA1', 'LA2', 'LAD1', 'LAD2']
    samples = 100000
    ts = np.random.random((len(channels), samples))

    freq = int(200 * 1e3)
    ampl = 500
    duration = 500

    stim, sham = [], []

    for i, channel in enumerate(channels):
        stim.append((channel, freq, ampl, duration, i * 2000, hashit(i)))
        sham.append((channel, freq, ampl, duration, (i + len(channels)) * 2000, hashit(i + len(channels))))

    stim = np.rec.array(stim, dtype=_events_dtype)
    sham = np.rec.array(sham, dtype=_events_dtype)

    detector = ArtifactDetector(ts, channels, stim, sham)
    print(detector.get_bad_channels())
