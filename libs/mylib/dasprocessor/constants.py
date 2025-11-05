"""
Module containing all constants.
"""

from warnings import warn

import numpy as np
import os

from .exceptions import TrialDataExistsError

_OWN_DIR = os.path.realpath(os.path.dirname(__file__))

properties = {
    "2024-05-03": {
        1: {
            "time_range": ((10, 59, 19), (11, 47, 29)),
            "offset_in_samples": 146000,
            "sample_rate": 25000,
            "gauge_length": 2.04,
            "transmitter_altitude": -5,
            "channel_distance": 1.02,
            "channel_count": 1200,
            "sequence_start": (10, 59, 24),
            "sequence_count": 128,
            "sequence_period": 562500,
            "cumulative_sequence_start": 0,
            "source_signal_window": (200*64000, 3200*64000),
            "source_position_file": "ellyandcable-firstrun.npz",
        },
        2: {
            "time_range": ((12, 12, 23), (13, 0, 33)),
            "offset_in_samples": 120000,
            "sample_rate": 25000,
            "gauge_length": 2.04,
            "transmitter_altitude": -30,
            "channel_distance": 1.02,
            "channel_count": 1200,
            "sequence_start": (12, 12, 26),
            "sequence_count": 128,
            "sequence_period": 562500,
            "cumulative_sequence_start": 128,
            "source_signal_window": (4500*64000, 7600*64000),
            "source_position_file": "ellyandcable-secondrun.npz",
        },
        3: {
            "time_range": ((13, 9, 50), (13, 47, 10)),
            "offset_in_samples": 118000,
            "sample_rate": 25000,
            "gauge_length": 4.08,
            "transmitter_altitude": {0: -35, 900: -25, 1080: -20, 1800: -15},
            "channel_distance": 2.04,
            "channel_count": 600,
            "sequence_start": (13, 9, 53),
            "sequence_count": 99,
            "sequence_period": 562500,
            "cumulative_sequence_start": 256,
            "source_signal_window": (8000*64000, 10400*64000),
            "source_position_file": "ellyandcable-thirdrun.npz",
        },
        "metadata": {
            "weather": "clear",
            "receiver_2_coordinates": (63.4406, 10.3518, -60),
            # above is lat-lon-alt
            "signal_sequence": ("B_4", "B_2", "B", "C", "A", "chirp"),
            "signal_starting_points": tuple(
                np.round(np.array([0, 9.3, 14.2, 16.9, 19.16, 20.76])
                         * 25000).astype('int64')),
            "sound_speed_profile_file": None,
        }
    }
}
"""Dictionary of trial data.

:meta hide-value:

Each trial day is a dictionary with integer keys corresponding to individual
runs and a ``"metadata"`` key that contains information about the trial day.
For example, an experiment carried out on January 1, 1970 and consisting of
four trial runs would have the structure ``{"1970-01-01": {1: ..., 2: ...,
3: ..., 4: ..., "metadata": ...}``.

Each trial run has a dictionary with the following keys:

*   **time_range**: 2-tuple of 3-tuples on the form (hh, mm, ss), compatible
    with :py:func:`dasprocessor.utils.to_time_list`.
*   **offset_in_samples**: how many samples should be skipped in the DAS data
    when processing.
*   **sample_rate**: DAS interrogator samples per second.
*   **gauge_length**: DAS interrogator gauge length, in metres.
*   **transmitter_altitude**: Altitude of transmitter. If the transducer depth
    changed during the run, it is a dict with the time in seconds since
    the start of the run as key. The most recent time offset applies.
*   **channel_distance**: How far apart in space the virtual channels are
    spaced.
*   **channel_count**: Number of virtual channels in the DAS data.
*   **sequence_count**: Count of signal sequences that were transmitted in this
    run.
*   **sequence_period**: Duration of one signal sequence in DAS interrogator
    samples.
*   **cumulative_sequence_start**: Number of signal sequences that have been
    transmitted before the start of this run.
*   **source_signal_window**: Requires a second receiver, e.g. a hydrophone.
    2-tuple on the form (start of slice containing relevant source data,
    stop of slice containing relevant source data), expressed in samples
    *on the second receiver*.
*   **source_position_file**: Name of the file containing source positional
    data. Must match exactly.
"""
frequency_bands = {
            "B_4": (1200, 1800),
            "B_2": (2400, 3600),
            "B": (4800, 7200),
            "C": (8200, 11200),
            "A": (9080, 13880),
            "chirp": (750, 15000),
        }
"""Bandpass filter cut-off frequencies per signal kind.

:meta hide-value:
"""

janus_frequency_bands = {
            "B_4": {"fmin": 1240, "df": 20},
            "B_2": {"fmin": 2480, "df": 40},
            "B": {"fmin": 4960, "df": 80},
            "C": {"fmin": 8400, "df": 100},
            "A": {"fmin": 9440, "df": 160},
        }
"""Frequency band properties of JANUS bands.

:meta hide-value:
"""

baseline_chips = 176
"""Number of chips in a JANUS baseline packet."""
janus_frequency_count = 26
"""Frequency bins in a JANUS packet, both zeroes and ones."""
hopping_pattern = np.loadtxt((_OWN_DIR + "/../resources/hop.txt")
                             ).astype('int32')[:, None]
"""Column vector with the hopping pattern.

:meta hide-value:
"""


def register_experiment(date, loaded_runs, loaded_meta, on_exist="raise"):
    """Register trial data.

    .. caution ::

        The function has not been tested yet.

    :param date: Date of experiment to register.
    :type date: date-like
    :param loaded_runs: Table with run data. Use
        :py:func:`dasprocessor.saveandload.load_experiment_run_table`
        to load them.
    :type loaded_runs: DataFrame
    :param loaded_meta: Experiment metadata. Use
        :py:func:`dasprocessor.saveandload.load_experiment_metadata`
        to load them.
    :type loaded_meta: Mappable
    :param on_exist: Behaviour of the function when an experiment
        already exists.

            'raise'
                Raise an exception. This is the default behaviour.

            'ignore'
                Do nothing unsuccessfully.

            'replace'
                Replace the already registered experiment.

            'ignore-existing'
                Merge the experiments. Existing records take precedence.

            'replace-existing'
                As 'ignore-existing', but lets new records take precedence.
    :type on_exist: {'raise', 'ignore', 'replace', 'ignore-existing',
        'replace-existing'}
    :returns: ``True`` if the registry of experiments changed,
        ``False`` otherwise.
    :raises TrialDataExistsError: if trial data exist for the specified
        date and are not ``None``, and the default behaviour is used.
    """
    existing_record = properties.get(date, None)
    # loaded_runs = load_experiment_run_table(path_runs)
    loaded_dict = {run: {k: loaded_runs.at[run, k]
                         for k in loaded_runs.columns}
                   for run in loaded_runs.index}
    # loaded_metadata = load_experiment_metadata(path_meta)
    if existing_record is None or on_exist == 'replace':
        # empty or don't care because we are replacing them anyway
        properties[date] = loaded_dict
        properties[date]['metadata'] = loaded_metadata
        return True

    match on_exist:
        case 'raise':
            raise TrialDataExistsError('there are already trial data '
                                       f'registered for {date}; use "replace"'
                                       ' or "replace-existing" to overwrite '
                                       'them, or "ignore" or "ignore-existing"'
                                       ' to not overwrite them')
        case 'ignore':
            return False
        case 'ignore-existing':
            for k in loaded_dict.keys() & properties[date].keys():
                # because keys are overwritten and I want to merge
                loaded_dict[k] |= properties[date][k]
                properties[date][k] = loaded_dict[k]

            if loaded_dict != properties[date]:
                warn(UserWarning("New records were added. If you are in an "
                                 "interactive prompt, confirm that the new "
                                 "run data are consistent with the old data."))
                return True
            return False
        case 'replace-existing':
            for k in loaded_dict.keys() & properties[date].keys():
                # because keys are overwritten and I want to merge
                properties[date][k] |= loaded_dict[k]

            if loaded_dict != properties[date]:
                warn(UserWarning("Some records were not replaced. If you are"
                                 " in an interactive prompt, confirm that the"
                                 " old run data are consistent with the new"
                                 " data."))

            return True
        case _:
            raise ValueError(f'unsupported on-exist behaviour {on_exist}')


def get_run(date: str, run: int) -> dict:
    """Fetch trial data from a single run and day.

    :param date: Date of experiment on the form ``yyyy-mm-dd``.
    :type date: date-like str
    :param run: Run number. The first run is always number 1.
    :type run: positive int
    :returns: Dictionary with properties of the chosen run.
        See documentation of :py:attr:`properties` for an explanation of the
        output.

    """
    return properties[date][run]


def get_trial_day_metadata(date: str) -> dict:
    """Fetch trial day metadata.

    :param date: Date of experiment on the form ``yyyy-mm-dd``.
    :type date: date-like str
    :returns: The trial metadata. See documentation of :py:attr:`properties`
        for an explanation of the output.

    """
    return properties[date]["metadata"]


def main():
    """Demonstrate that constants can be loaded."""
    print(get_run("2024-05-03", int(input("Choose a run between 1 and 3.\n"))))


if __name__ == "__main__":
    main()
