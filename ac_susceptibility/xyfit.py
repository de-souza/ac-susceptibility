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
from scipy.optimize import least_squares


def xyfit(data, calibration_data):
    """Return the fit of the voltage data and its parameters.

    Args:
        data: An array containing the position of the sample, the
            amplitude of the voltage measured by the lock-in amplifier,
            and its phase.

    Returns:
        A tuple of containing the fitted data and a list of the
        amplitude of the baseline and each peak and their phase.

    """
    if "fit_parameters" in calibration_data:
        x_params, y_params = calibration_data["fit_parameters"]
        x_fit, x_pfit = partial_fit(data[:, 0], data[:, 1], x_params)
        y_fit, y_pfit = partial_fit(data[:, 0], data[:, 2], y_params)
    else:
        x_fit, x_pfit = complete_fit(data[:, 0], data[:, 1])
        y_fit, y_pfit = complete_fit(data[:, 0], data[:, 2])

    z_fit = x_fit + 1j * y_fit
    pfit = x_pfit + 1j * y_pfit

    fit = np.empty_like(data)
    fit[:, 0] = data[:, 0]
    fit[:, 1] = x_fit
    fit[:, 2] = y_fit
    fit[:, 3] = np.abs(z_fit)
    fit[:, 4] = np.angle(z_fit, deg=True)

    return fit, pfit


def complete_fit(position, voltage):
    """Return a asymmetric double sigmoidal fit and its parameters.

    Args:
        position: An array of floats of the position.
        voltage: An array of floats of the voltage.

    """
    pos_max = position[voltage.argmax()]
    pos_min = position[voltage.argmin()]
    baseline = (max(voltage) + min(voltage)) / 2
    u_0 = baseline
    u_max = max(voltage) - baseline
    u_min = min(voltage) - baseline
    x_c1 = pos_max - (pos_min - pos_max) / 4
    x_c2 = pos_max + (pos_min - pos_max) / 4
    x_c3 = pos_min - (pos_min - pos_max) / 4
    x_c4 = pos_min + (pos_min - pos_max) / 4
    w_1 = 2
    w_2 = 2
    w_3 = 2
    w_4 = 2
    init_params = [u_0, u_max, u_min, x_c1, x_c2, x_c3, x_c4, w_1, w_2, w_3, w_4]

    residuals = lambda pfit: voltage - asym2sig(position, pfit)

    pfit = least_squares(residuals, init_params).x
    fit = asym2sig(position, pfit)

    return fit, pfit


def partial_fit(position, voltage, fixed_params):
    """Return a asymmetric double sigmoidal fit and its parameters.

    Args:
        position: An array of floats of the position.
        voltage: An array of floats of the voltage.
        fixed_params: A list of floats of the known parameters.

    """
    x_c1 = lambda x_c: fixed_params[0] + x_c
    x_c2 = lambda x_c: fixed_params[1] + x_c
    x_c3 = lambda x_c: fixed_params[2] + x_c
    x_c4 = lambda x_c: fixed_params[3] + x_c
    w_1 = fixed_params[4]
    w_2 = fixed_params[5]
    w_3 = fixed_params[6]
    w_4 = fixed_params[7]
    partial_params = lambda u_0, u_max, u_min, x_c: np.array([
        u_0,
        u_max,
        u_min,
        x_c1(x_c),
        x_c2(x_c),
        x_c3(x_c),
        x_c4(x_c),
        w_1,
        w_2,
        w_3,
        w_4,
    ])
    partial_asym2sig = lambda pos, params: asym2sig(pos, partial_params(*params))

    baseline = (max(voltage) + min(voltage)) / 2
    init_u_0 = baseline
    init_u_max = max(voltage) - baseline
    init_u_min = min(voltage) - baseline
    init_x_c = 0
    init_params = [init_u_0, init_u_max, init_u_min, init_x_c]

    residuals = lambda params: voltage - partial_asym2sig(position, params)

    pfit = least_squares(residuals, init_params).x
    fit = partial_asym2sig(position, pfit)

    return fit, partial_params(*pfit)


def asym2sig(pos, params):
    """Return an asymmetric double sigmoidal function of the position.

    The asymmetric double sigmoidal function is the difference of two
    independent sigmoidal functions. They are expressed here as 
    hyperbolic tangents in order to reduce the computer time.

    Args:
        pos: A float or array of floats of the position.
        params: A list of all the function parameters.

    """
    u_0, u_max, u_min, x_c1, x_c2, x_c3, x_c4, w_1, w_2, w_3, w_4 = params
    return (
        u_0
        + u_max * (np.tanh((pos - x_c1) / w_1) - np.tanh((pos - x_c2) / w_2)) / 2
        + u_min * (np.tanh((pos - x_c3) / w_3) - np.tanh((pos - x_c4) / w_4)) / 2
    )
