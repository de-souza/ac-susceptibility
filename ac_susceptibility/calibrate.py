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


def calibrate(data_path):
    """Calibrate.

    Args:
        data_path: A data folder as a "Path" object.

    """
    calibration_data = {
        "fit_parameters": [
            [
                -1.31482959e01,
                -1.87763231e01,
                -2.52376225e01,
                -3.07832386e01,
                2.66677577e00,
                2.75115124e00,
                2.52019421e00,
                3.20146748e00,
            ],
            [
                -1.31740494e01,
                -1.87372536e01,
                -2.51468752e01,
                -3.08226837e01,
                2.77568482e00,
                2.87184774e00,
                2.50477186e00,
                3.11997439e00,
            ],
        ]
    }
    return calibration_data
