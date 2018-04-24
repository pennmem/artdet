# EEG artifact detection

Determine saturated and artifactual EEG channels to exclude from further
analyses. 

## Algorithm description

The algorithm consists of two independent steps:

1. Saturation detection
2. Artifact detection

### Saturation detection

Channels are rejected if the nth discrete difference of power computed during a
post-stim interval is above a threshold value.

By default, `n = 10` and `threshold = 10`.

### Artifact detection

The following procedure is used in an experiment:

* Stimulation is applied `N_stim` times per stim channel
* `N_sham` sham events are also applied

By default, `N_stim = N_sham = 30`.

Following all stim and sham events:

* Compute average signal for each pre- and post-stim interval
* Compute post-pre difference `d_stim` (`d_sham`) for stim (sham) intervals
* Compute standard deviation `s_sham` of `d_sham`
* Identify stim events where `d_stim` is outside of &pm;`3 * s_sham`
* If more than 30% of events above are found, identify the channel as exhibiting
  excessive artifact
  
By default, pre-stim interval is -440 to -40 ms relative to stim onset and
post-stim interval is 40 to 440 ms.
