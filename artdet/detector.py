"""Rejection of artifactual and saturated EEG channels."""

from collections import namedtuple

import numpy as np
from traits.api import HasTraits, Array, Int

# Container for results from artifact detection
ArtifactDetectionResults = namedtuple("ArtifactDetectionResults", "saturated,artifactual,mask")


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

    def __init__(self, pre_intervals, post_intervals, sham_pre_intervals,
                 sham_post_intervals, saturation_order=None,
                 saturation_threshold=None):
        super(ArtifactDetector, self).__init__()

        assert pre_intervals.shape == post_intervals.shape
        assert len(pre_intervals.shape) == 3

        assert sham_pre_intervals.shape == sham_post_intervals.shape
        assert len(sham_pre_intervals) == 3
        assert sham_pre_intervals.shape[2] == pre_intervals.shape[2]

        self.pre_intervals = pre_intervals
        self.post_intervals = post_intervals
        self.sham_pre_intervals = sham_pre_intervals
        self.sham_post_intervals = sham_post_intervals

        if saturation_order is not None:
            self.saturation_order = saturation_order

        if saturation_threshold is not None:
            self.saturation_threshold = saturation_threshold

    def get_saturated_channels(self):
        """Identify channels which display post-stim saturation.

        Returns
        -------
        mask : np.ndarray
            Boolean mask over the channel axis to indicate channels exhibiting
            saturation.

        """
        time_axis = 2
        deriv = np.diff(self.post_intervals,
                        n=self.saturation_order,
                        axis=time_axis)
        mask = ((deriv == 0).sum(time_axis) > self.saturation_threshold).any(0).squeeze()
        return mask

    def get_artifactual_channels(self):
        """Identify channels which display significant post-stim artifact. See
        :meth:`get_saturated_channels` for return value info.

        Notes
        -----
        Stimulation channels are not treated any differently by
        this method.

        """
        time_axis = 2
        n_events = float(self.pre_intervals.shape[0])

        # mean signals over time
        m_pre_stim = self.pre_intervals.mean(axis=time_axis)
        m_post_stim = self.post_intervals.mean(axis=time_axis)
        m_pre_sham = self.sham_pre_intervals.mean(axis=time_axis)
        m_post_sham = self.sham_post_intervals.mean(axis=time_axis)

        # post - pre deltas
        d_stim = m_post_stim - m_pre_stim
        d_sham = m_post_sham - m_pre_sham

        # standard deviations over channels
        s_sham = d_sham.std(axis=0)

        # identify outlier events
        outliers = d_stim >= 3 * s_sham

        # mark channels with >= 30% outliers
        mask = outliers.sum(axis=0).astype(np.float) / n_events >= 0.3
        return mask

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
