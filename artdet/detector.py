"""Rejection of artifactual and saturated EEG channels."""

from collections import namedtuple
import random

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

    """
    pre_intervals = Array(desc='pre-stim interval array')
    post_intervals = Array(desc='post-stim interval array')

    saturation_order = Int(10, desc='derivative order')
    saturation_threshold = Int(10, desc='number of points where order derivative is equal to zero')

    def __init__(self, pre_intervals, post_intervals, saturation_order=None,
                 saturation_threshold=None):
        super(ArtifactDetector, self).__init__()

        assert pre_intervals.shape == post_intervals.shape
        assert len(pre_intervals.shape) == 3
        self.pre_intervals = pre_intervals
        self.post_intervals = post_intervals

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
        :meth:`_get_saturated_channels` for return value info.

        Notes
        -----
        Stimulation channels are not treated any differently by
        this method.

        """
        # TODO
        return np.array([
            random.choice([True, False])
            for _ in range(self.post_intervals.shape[1])
        ])

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
