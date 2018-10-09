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
from pathlib import Path
import numpy as np


def organize(data_path):
    """Reorganize data in temperature subfolders and frequency files.

    Args:
        data_path: A data folder as a "Path" object.

    """
    input_folder = data_path / "input"
    for measurement_folder in input_folder.iterdir():
        for unsorted_subfolder in iter_unsorted_subfolders(measurement_folder):

            remove_non_txt_files(unsorted_subfolder)

            measurement_numbers = {
                get_measurement_number(file) for file in unsorted_subfolder.iterdir()
            }

            for number in measurement_numbers:

                unsorted_files = [
                    file
                    for file in unsorted_subfolder.iterdir()
                    if get_measurement_number(file) == number
                ]

                sorted_subfolder = measurement_folder / get_temperature(unsorted_files)
                sorted_subfolder.mkdir(parents=True, exist_ok=True)

                for file in unsorted_files:
                    sorted_file = sorted_subfolder / sorted_filename(file)
                    file.rename(sorted_file)

            unsorted_subfolder.rmdir()


def iter_unsorted_subfolders(folder):
    """Return an iterator over not yet sorted subfolders.

    Args:
        folder: A folder as a "Path" object.

    """
    return (
        subfolder for subfolder in folder.iterdir() if not subfolder.name.endswith("K")
    )


def remove_non_txt_files(folder):
    """Delete all non-text files within a folder.

    Args:
        folder: A folder as a "Path" object.

    """
    for file in folder.iterdir():
        if not file.name.endswith(".txt"):
            file.unlink()


def get_measurement_number(file):
    """Return a measurement number from a file's name.

    Args:
        file: A file as a "Path" object.

    """
    return int(file.stem[-4:])


def get_temperature(files):
    """Return a string of the mean temperature from an list of files.

    Args:
        files: An iterable of files as "Path" objects.

    """
    temperature_list = [
        np.nanmean(np.genfromtxt(file.as_posix(), skip_header=5, usecols=(6)))
        for file in files
    ]
    temperature = np.around(np.nanmean(temperature_list))
    return "{:g}K".format(temperature)


def sorted_filename(file):
    """Return the organized version of a file name.

    Args:
        file: A file as a "Path" object.

    """
    return file.stem[12:-5] + ".txt"
