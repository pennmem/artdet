import os

import h5py
import numpy as np
from numpy.testing import assert_equal
import pytest

from cmlreaders.path_finder import PathFinder
from cmlreaders.ramulator import events_to_dataframe

from artdet.detector import ArtifactDetector


def make_path_finder():
    rootdir = os.path.expanduser(os.environ.get('RHINO_ROOT', '/'))
    finder = PathFinder(subject='R1395M', experiment='catFR5', session=4,
                        rootdir=rootdir)
    return finder


def load_test_data():
    """Loads test data and pre-processes into pre- and post-stim intervals.

    Returns a dict with keys 'pre' and 'post'.

    TODO: include a small dataset to test on TravisCI.

    """
    finder = make_path_finder()
    event_log = finder.find_file('event_log')
    hfile_path = finder.find_file('raw_eeg')

    with h5py.File(hfile_path, 'r') as hfile:
        events = events_to_dataframe(event_log)

        # skip the firststim stim events
        firststim = 15

        # only using 30 events because this is how many we are going to use by
        # default (per channel) in Ramulator
        stim_events = events[events['event_label'] == 'STIM'].dropna(axis=1)[firststim:(firststim + 30)]

        mask = (
            (events.event_label == 'WORD') &
            (events.event_value == True) &
            (events['msg_stub.data.phase_type'].isin(['BASELINE', 'PRACTICE']))
        )
        sham_events = events[mask][:30]

        # FIXME: make sham events during baseline encoding

        # maximum channel index to read data from (-1 to load all)
        maxchan = 32

        # transposing data such that we are shaped with axes
        # (events, channels, time) which is what Leon's algorithm requires for
        # saturation detection
        prestim = np.array([
            hfile['/timeseries'][(start - 440):(start - 40), :maxchan].T
            for start in stim_events.offset
        ])
        poststim = np.array([
            hfile['/timeseries'][(start + 40):(start + 440), :maxchan].T
            for start in stim_events.offset + 500
        ])
        presham = np.array([
            hfile['/timeseries'][(start - 440):(start - 40), :maxchan].T
            for start in sham_events.offset
        ])
        postsham = np.array([
            hfile['/timeseries'][(start + 40):(start + 440), :maxchan].T
            for start in sham_events.offset + 500
        ])

    return {
        'pre': prestim,
        'post': poststim,
        'presham': presham,
        'postsham': postsham,
    }


class TestArtifactDetector:
    @classmethod
    def setup_class(cls):
        data = load_test_data()
        pre = data['pre']
        post = data['post']
        pre_sham = data['presham']
        post_sham = data['postsham']
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