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

from .load import load_file
from .xyfit import xyfit


def calibrate(data_path):
    """Obtain the calibration values from a set of coils.

    Args:
        data_path: A data folder as a "Path" object.

    """
    calibration_data = {}

    fit_params = fit_parameters(data_path)
    if fit_params is not None:
        calibration_data["fit_parameters"] = fit_params.real, fit_params.imag

    return calibration_data


def fit_parameters(data_path):
    """Obtain the best values for the fit parameters.

    Args:
        data_path: A data folder as a "Path" object.

    """
    folder_path = data_path / "calibration" / "fit_parameters"
    files = list(folder_path.iterdir())

    if files:
        pfit = np.empty((len(files), 11), dtype=np.complex_)
        for i, voltage_file in enumerate(files):
            data = load_file(voltage_file)
            _, pfit[i] = xyfit(data, {})
        fit_params = np.median(pfit, axis=0)[3:]
    else:
        fit_params = None

    return fit_params
