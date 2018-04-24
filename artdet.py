"""Rejection of artifactual and saturated channels."""

import random

import numpy as np
import pandas as pd
from traits.api import HasTraits, Array,ListStr


class ArtifactDetector(HasTraits):
    """Class used to detect artifactual and saturated channels."""
    timeseries = Array(desc='time series data, shape = (n_channels, n_samples)')
    channels = ListStr(desc='channel labels')
    stim_events = Array(desc='stimulation events')
    sham_events = Array(desc='sham stimulation events')

    def __init__(self, timeseries, channels, stim_events, sham_events):
        super(ArtifactDetector, self).__init__()

        self.timeseries = timeseries
        self.channels = channels
        self.stim_events = stim_events
        self.sham_events = sham_events

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
    channels = ['LA1', 'LA2', 'LAD1', 'LAD2']
    samples = 10000
    ts = np.random.random((len(channels), samples))
    stim = np.array([1, 2, 3])
    sham = np.array([1, 2, 3])

    detector = ArtifactDetector(ts, channels, stim, sham)
    print(detector.get_bad_channels())
