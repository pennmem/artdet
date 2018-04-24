"""Rejection of artifactual and saturated channels."""

import random

import numpy as np
import pandas as pd
from traits.api import HasTraits, Array, CArray, ListStr

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
    pre_stim_interval = CArray(value=[-440, -40], dtype=np.int, shape=(2,),
                               desc='pre-stim start and stop times')
    post_stim_interval = CArray(value=[40, 440], dtype=np.int, shape=(2,),
                                desc='post-stim start and stop times')

    def __init__(self, timeseries, channels, stim_events, sham_events,
                 pre_stim_interval=None, post_stim_interval=None):
        super(ArtifactDetector, self).__init__()

        self.timeseries = timeseries
        self.channels = channels
        self.stim_events = stim_events
        self.sham_events = sham_events

        if pre_stim_interval is not None:
            self.pre_stim_interval = pre_stim_interval

        if post_stim_interval is not None:
            self.post_stim_interval = post_stim_interval

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
            dimensions (n_events, n_channels, n_samples_in_interval)

        """
        events = self.sham_events if sham else self.stim_events

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
    samples = 10000
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
