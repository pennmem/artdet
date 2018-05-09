"""Rejection of artifactual and saturated EEG channels."""

from collections import namedtuple

import numpy as np
from traits.api import HasTraits, Array, Int, Float

# Container for results from artifact detection
ArtifactDetectionResults = namedtuple("ArtifactDetectionResults",
                                      "saturated,artifactual,mask")


class ArtifactDetector(HasTraits):
    """Class used to detect artifactual and saturated channels.

    Parameters
    ----------
    pre_intervals : np.ndarray
        Pre-stim intervals as a (n_events x n_channels x time) array
    post_intervals : np.ndarray
        Post-stim intervals as a (n_events x n_channels x time) array
    sham_pre_intervals : np.ndarray
        Sham pre-stim intervals
    sham_post_intervals : np.ndarray
        Sham post-stim intervals

    """
    pre_intervals = Array(desc='pre-stim interval array')
    post_intervals = Array(desc='post-stim interval array')
    sham_pre_intervals = Array(desc='sham pre-stim interval array')
    sham_post_intervals = Array(desc='sham post-stim interval array')

    saturation_order = Int(10, desc='derivative order')
    saturation_threshold = Int(10, desc='number of points where order derivative is equal to zero')

    artifactual_sd = Int(3, desc='standard deviations from mean to consider event artifactual')
    artifactual_ratio = Float(0.5, desc='fraction of events required to be over threshold to flag a channel as bad')

    def __init__(self, pre_intervals, post_intervals, sham_pre_intervals,
                 sham_post_intervals, saturation_order=None,
                 saturation_threshold=None, artifactual_sd=None,
                 artifactual_ratio=None):
        super(ArtifactDetector, self).__init__()

        assert pre_intervals.shape == post_intervals.shape
        assert len(pre_intervals.shape) == 3

        assert sham_pre_intervals.shape == sham_post_intervals.shape
        assert len(sham_pre_intervals.shape) == 3
        assert sham_pre_intervals.shape[2] == pre_intervals.shape[2]

        self.pre_intervals = pre_intervals
        self.post_intervals = post_intervals
        self.sham_pre_intervals = sham_pre_intervals
        self.sham_post_intervals = sham_post_intervals

        if saturation_order is not None:
            self.saturation_order = saturation_order

        if saturation_threshold is not None:
            self.saturation_threshold = saturation_threshold

        if artifactual_sd is not None:
            self.artifactual_sd = artifactual_sd

        if artifactual_ratio is not None:
            self.artifactual_ratio = artifactual_ratio

        self.event_axis = 0
        self.channel_axis = 1
        self.time_axis = 2

    def get_saturated_channels(self):
        """Identify channels which display post-stim saturation.

        Returns
        -------
        mask : np.ndarray
            Boolean mask over the channel axis to indicate channels exhibiting
            saturation.

        """
        deriv = np.diff(self.post_intervals,
                        n=self.saturation_order,
                        axis=self.time_axis)
        mask = (((deriv == 0).sum(self.time_axis) >
                 self.saturation_threshold).any(0).squeeze())

        return mask

    def get_artifactual_channels(self, method='zscore'):
        """Identify channels which display significant post-stim artifact.

        Parameters
        ----------
        method : str
            One of: ``zscore``, ``tstat`` (default: ``zscore``)

        """
        if method == 'zscore':
            return self.get_artifactual_channels_by_zscore()
        elif method == 'tstat':
            return self.get_artifactual_channels_by_tstat()
        else:
            raise RuntimeError("Invalid bad channel detection method")

    def get_artifactual_channels_by_zscore(self):
        """Identify channels which display significant post-stim artifact using
        a zscore-based method.

        """
        n_events = float(self.pre_intervals.shape[self.event_axis])

        # mean signals over time
        m_pre_stim = self.pre_intervals.mean(axis=self.time_axis)
        m_post_stim = self.post_intervals.mean(axis=self.time_axis)
        m_pre_sham = self.sham_pre_intervals.mean(axis=self.time_axis)
        m_post_sham = self.sham_post_intervals.mean(axis=self.time_axis)

        # post - pre deltas
        d_stim = m_post_stim - m_pre_stim
        d_sham = m_post_sham - m_pre_sham

        # mean and standard deviation of the sham data by channel (over events)
        s_sham = d_sham.std(axis=self.event_axis)
        m_sham = d_sham.mean(axis=self.event_axis)

        # identify outlier events using the mean and sd of the sham post-pre
        # difference as the normalization factor
        outliers = np.abs((d_stim - m_sham) / s_sham) >= self.artifactual_sd

        # mark channels with a proportion of events marked as artifactual over
        # the given threshold
        mask = (outliers.sum(axis=self.event_axis).astype(np.float) /
                n_events >= self.artifactual_ratio)

        return mask

    def get_artifactual_channels_by_tstat(self):
        """Identify channels which display significant post-stim artifact using
        a t-stat-based method.

        """
        raise NotImplementedError

    def get_bad_channels(self):
        """Identify all bad channels.

        Returns
        -------
        saturated : np.ndarray
            Boolean mask of all channels rejected due to saturation
        artifactual : np.ndarray
            Boolean mask of all channels rejected due to excessive artifact
        mask : np.ndarray
            Combined boolean mask (logical or of both the above)

        """
        saturated = self.get_saturated_channels()
        artifactual = self.get_artifactual_channels()
        mask = np.logical_or(saturated, artifactual)
        return ArtifactDetectionResults(saturated, artifactual, mask)


if __name__ == "__main__":
    sample_pre_stim = np.random.normal(0, 3, (100, 50, 30))
    sample_post_stim = np.random.normal(0, 4, (100, 50, 30))
    sample_pre_sham = np.random.normal(0, 3, (100, 50, 30))
    sample_post_sham = np.random.normal(0, 3, (100, 50, 30))

    detector = ArtifactDetector(sample_pre_stim, sample_post_stim,
                                sample_pre_sham, sample_post_sham)
    artifactual_channels = detector.get_artifactual_channels()
    saturated_channels = detector.get_saturated_channels()
