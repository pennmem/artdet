import numpy as np
from numpy.testing import assert_equal

from artdet.data import load_data
from artdet.detector import ArtifactDetector


def load_test_data():
    """Loads test data and pre-processes into pre- and post-stim intervals.

    Returns a dict with keys 'pre' and 'post'.

    TODO: include a small dataset to test on TravisCI.

    """
    return load_data('R1395M', 'catFR5', 4, maxchan=32)


class TestArtifactDetector:
    @classmethod
    def setup_class(cls):
        data = load_test_data()
        pre = data['pre-stim']
        post = data['post-stim']
        pre_sham = data['pre-sham']
        post_sham = data['post-sham']
        cls.detector = ArtifactDetector(pre, post, pre_sham, post_sham)

    def test_get_saturated_channels(self):
        mask = self.detector.get_saturated_channels()

        # FIXME: explicitly check that this gives the right results
        assert not all(mask)

    def test_get_artifactual_channels(self):
        mask = self.detector.get_artifactual_channels()

        # FIXME: explicitly check that this gives the right results
        assert not all(mask)

    def test_get_bad_channels(self):
        masks = self.detector.get_bad_channels()
        assert len(masks) == 3
        assert_equal(np.logical_or(masks[0], masks[1]), masks[2])
