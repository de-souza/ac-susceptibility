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
import argparse

from .organize import organize
from .plot import plot


def ac_susceptibility(skip_voltage, data_path):
    """Organize and plot ac-susceptibility measurement data."""
    if data_path is None:
        data_path = Path(__file__).parents[1] / "data"
    else:
        data_path = Path(data_path)

    organize(data_path)

    plot(data_path, skip_voltage)


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Organize and plot ac-susceptibility measurement data."
    )
    parser.add_argument(
        "-s", "--skip-voltage", action="store_true", help="don't plot the voltage"
    )
    parser.add_argument("-d", "--data-path", help="path to data folder")
    args = parser.parse_args()
    return args.skip_voltage, args.data_path


if __name__ == "__main__":
    ac_susceptibility(*parse_args())
