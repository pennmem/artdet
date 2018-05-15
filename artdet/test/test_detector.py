import numpy as np
from numpy.testing import assert_equal
import pytest

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

    def test_ttest_method(self):
        self.detector.get_artifactual_channels_by_ttest()

    @pytest.mark.parametrize('method', ['zscore', 'ttest', 'notreal'])
    def test_get_bad_channels2(self, method):
        if method == 'notreal':
            with pytest.raises(RuntimeError):
                self.detector.get_bad_channels(method=method)
            return

        sat, art, mask = self.detector.get_bad_channels(method=method)
        assert isinstance(mask, np.ndarray)
        assert isinstance(art, np.ndarray)
        assert isinstance(sat, np.ndarray)
        assert mask.shape == art.shape == sat.shape


if __name__ == "__main__":
    from contextlib import contextmanager
    from functools import partial
    import os

    import h5py

    get_path = partial(os.path.join, os.environ['RHINO_ROOT'], 'scratch', 'depalati')
    inpath = get_path('artdet.h5')
    outpath = get_path('artdet_results3.h5')

    @contextmanager
    def open_hdf5_files():
        with h5py.File(inpath, 'r') as infile:
            with h5py.File(outpath, 'w') as outfile:
                yield infile, outfile

    with open_hdf5_files() as files:
        infile, outfile = files
        count = infile.attrs['num_datasets']

        for i in range(count):
            print("analyzing dataset", i)
            data = infile['dataset_{}'.format(i)]

            try:
                detector = ArtifactDetector(data['pre-stim'][:], data['post-stim'][:],
                                            data['pre-sham'][:], data['post-sham'][:],
                                            artifactual_sd=2, artifactual_ratio=0.5)
            except:
                print("error, moving on")
                continue

            saturated, artifactual, mask = detector.get_bad_channels()

            group = outfile.create_group('dataset_{}'.format(i))
            save = partial(group.create_dataset, chunks=True)
            save('saturated', data=saturated)
            save('artifactual', data=artifactual)
            save('mask', data=mask)
