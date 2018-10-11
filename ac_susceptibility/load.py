#
#    Copyright 2018 Léo De Souza
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import numpy as np


def load_file(file):
    """Return the position, amplitude and phase data from a file.

    Args:
        file: A file's "Path" object.

    Returns:
        An array containing the position of the sample, the amplitude of
        the voltage measured by the lock-in amplifier, and its phase.

    """
    data = np.genfromtxt(file.as_posix(), skip_header=5, usecols=(0, 1, 2, 3, 4))
    data[:, 0] -= data[0, 0]  # zero as first position
    data[:, 0] /= 250  # to millimeters
    return data


def list_subfolders(folder):
    """Return list of subfolders sorted by temperature.

    Args:
        folder: A parent folder as a "Path" object.

    Returns:
        A list of "Path" objects corresponding to the sorted temperature
        subfolders.

    """
    subfolders = [entry for entry in folder.iterdir() if entry.is_dir()]
    subfolders = sorted(subfolders, key=lambda x: int(x.name.split("K")[0]))
    return subfolders


def list_freqs_and_files(folder):
    """Return list of "fequency, file" tuples within a folder.

    Args:
        folder: A parent folder as a "Path" object.

    Returns:
        A list of (freq, file) tuples, where 'freq' is a string of the
        frequency and 'file' is the file's 'Path' object.

    """
    files = [entry for entry in folder.iterdir() if entry.name.endswith("txt")]
    freqs = [float(entry.stem[:-2]) for entry in files]
    return sorted(zip(freqs, files))
