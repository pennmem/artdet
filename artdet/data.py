import os

import h5py
import numpy as np

from cmlreaders.path_finder import PathFinder
from cmlreaders.ramulator import events_to_dataframe


def make_path_finder(subject, experiment, session, rootdir=None):
    """Create the path finder using the ``RHINO_ROOT`` environment variable if
    no root directory is given.

    Parameters
    ----------
    subject : str
    experiment : str
    session : int

    """
    if rootdir is None:
        rootdir = os.path.expanduser(os.environ.get('RHINO_ROOT', '/'))
    finder = PathFinder(subject=subject, experiment=experiment, session=session,
                        rootdir=rootdir)
    return finder


def load_data(subject, experiment, session, firststim=15, maxchan=-1):
    """Loads test data and pre-processes into pre- and post-stim intervals.

    This assumes an FR5-like experiment.

    Parameters
    ----------
    subject : str
    experiment : str
        One of ``FR5``, ``catFR5``
    session : int

    Keyword arguments
    -----------------
    firststim : int
        Skips the first this many events
    maxchan : int
        Maximum number of channels to load (-1, the default, to load all)

    Returns
    -------
    dict with keys ``pre-stim``, ``post-stim``, ``pre-sham``, ``post-sham``
    containing 3D arrays of intervals with axes (events, channels, time).

    """
    finder = make_path_finder(subject, experiment, session)
    event_log = finder.find_file('event_log')
    hfile_path = finder.find_file('raw_eeg')

    with h5py.File(hfile_path, 'r') as hfile:
        events = events_to_dataframe(event_log)

        # only using 30 events because this is how many we are going to use by
        # default (per channel) in Ramulator
        stim_events = events[events['event_label'] == 'STIM'].dropna(axis=1)[firststim:(firststim + 30)]

        mask = (
            (events.event_label == 'WORD') &
            (events.event_value == True) &
            (events['msg_stub.data.phase_type'].isin(['BASELINE', 'PRACTICE']))
        )
        sham_events = events[mask][:30]

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
        'pre-stim': prestim,
        'post-stim': poststim,
        'pre-sham': presham,
        'post-sham': postsham,
    }
