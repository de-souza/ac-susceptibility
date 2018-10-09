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
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def plot(data_path, skip_voltage):
    """Plot magnetization and/or voltage from measurement files."""
    init_matplotlib()

    input_folder = data_path / "input"
    output_folder = data_path / "output"

    for measurement_folder in input_folder.iterdir():

        magnetization_data = []

        for temperature_folder in sorted_subfolders(measurement_folder):

            freqs_and_files = list_freqs_and_files(temperature_folder)
            temperature_data = np.empty((len(freqs_and_files), 7))

            for i, (freq, voltage_file) in enumerate(freqs_and_files):

                data = load_data(voltage_file)
                fit, pfit = fit_voltage(data)
                temperature_data[i] = freq, *pfit

                if not skip_voltage:
                    voltage_plot_path = (
                        output_folder
                        / "voltage"
                        / measurement_folder.name
                        / temperature_folder.name
                        / (voltage_file.stem + ".png")
                    )
                    make_voltage_plot(data, fit, voltage_plot_path)

            magnetization_data.append([temperature_folder.name, temperature_data])

        magnetization_plot_path = (
            output_folder / "magnetization" / (measurement_folder.name + ".pdf")
        )
        make_magnetization_plot(magnetization_data, magnetization_plot_path)


def init_matplotlib():
    """Set plot style."""
    plt.style.use("seaborn-notebook")
    plt.rc("lines", linewidth=1)
    plt.rc("grid", linewidth=0.1)


def sorted_subfolders(folder):
    """Return list of subfolders sorted by temperature.

    Args:
        folder: A parent folder as a "Path" object.

    Returns:
        A list of "Path" objects corresponding to the sorted temperature
        subfolders.

    """
    subfolders = [entry for entry in folder.iterdir() if entry.is_dir()]
    subfolders = sorted(subfolders, key=lambda x: int(x.name[:-1]))
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


def load_data(file):
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


def fit_voltage(data):
    """Return the fit of the voltage and its parameters.

    Args:
        data: An array containing the position of the sample, the
            amplitude of the voltage measured by the lock-in amplifier,
            and its phase.

    Returns:
        A tuple of containing the fitted data and a list of the
        amplitude of the baseline and each peak and their phase.

    """
    x_fit, x_pfit = _fit_asym2sig(data[:, 0], data[:, 1])
    y_fit, y_pfit = _fit_asym2sig(data[:, 0], data[:, 2])

    z_fit = x_fit + 1j * y_fit

    fit = np.empty_like(data)
    fit[:, 0] = data[:, 0]
    fit[:, 1] = x_fit
    fit[:, 2] = y_fit
    fit[:, 3] = np.abs(z_fit)
    fit[:, 4] = np.angle(z_fit, deg=True)

    z_pfit = x_pfit + 1j * y_pfit

    pfit = *np.abs(z_pfit), *np.angle(z_pfit, deg=True)

    return fit, pfit


def make_voltage_plot(data, fit, path):
    """Plot amplitude and phase of the signal versus sample position.

    Args:
        data: An array containing the position of the sample, the
            amplitude of the voltage measured by the lock-in amplifier,
            and its phase.
        fit: Same object as "data" with fitted values.
        path: The path of the saved plot as a "Path" object.

    """
    print('Saving plot "{}"...'.format(path))

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    _draw_ax(
        ax1,
        data[:, 0],
        data[:, 1] * 1000,
        fit[:, 1] * 1000,
        xlabel="Position (mm)",
        ylabel="X Channel (mV)",
    )

    _draw_ax(
        ax2,
        data[:, 0],
        data[:, 2] * 1000,
        fit[:, 2] * 1000,
        xlabel="Position (mm)",
        ylabel="Y Channel (mV)",
    )

    fig.tight_layout()
    fig.savefig(path.as_posix(), dpi=150)
    plt.close()


def make_magnetization_plot(data, path):
    """Plot amplitude and phase of the signal versus frequency.

    Args:
        data: An array or a list of arrays containing tuples of:
            - A string or list of strings containing the temperature.
            - The position of the sample, the amplitude of the baseline
                and each peak, and their phase.
        path: The path of the saved plot as a "Path" object.

    """
    print('Saving plot "{}"...'.format(path.name))

    if not isinstance(data[0], list):
        print("data is not a list")
        data = [data]

    temp = []
    curves = [[], [], [], [], [], [], []]

    for j, i in data:
        temp.append(j)
        curves[0].append(i[:, 0])
        curves[1].append(i[:, 1] / i[:, 0])
        curves[2].append(i[:, 2] / i[:, 0])
        curves[3].append(i[:, 3] / i[:, 0])
        # curves[2].append(i[:, 2] / i[:, 1])
        # curves[3].append(i[:, 3] / i[:, 1])
        curves[4].append(i[:, 4])
        curves[5].append(i[:, 5])
        curves[6].append(i[:, 6])

    max_baseline = max([i[0] for i in curves[1]])
    max_peaks = max(
        [np.abs(i[0]) for i in curves[2]] + [np.abs(i[0]) for i in curves[3]]
    )

    for i in range(len(data)):
        curves[1][i] /= max_baseline
        curves[2][i] /= max_peaks
        curves[3][i] /= max_peaks

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(12, 12))

    _draw_ax(
        ax1,
        curves[0],
        curves[1],
        title="Amplitude Baseline",
        xlabel="Frequency (Hz)",
        ylabel="Amplitude / Frequency (AU)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax2,
        curves[0],
        curves[4],
        title="Phase Baseline",
        xlabel="Frequency (Hz)",
        ylabel="Phase (°)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax3,
        curves[0],
        curves[2],
        title="Amplitude Peak #1",
        xlabel="Frequency (Hz)",
        ylabel="Amplitude / Frequency (AU)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax4,
        curves[0],
        curves[5],
        title="Phase Peak #1",
        xlabel="Frequency (Hz)",
        ylabel="Phase (°)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax5,
        curves[0],
        curves[3],
        title="Amplitude Peak #2",
        xlabel="Frequency (Hz)",
        ylabel="Amplitude / Frequency (AU)",
        xscale="log",
        legend=temp,
    )

    _draw_ax(
        ax6,
        curves[0],
        curves[6],
        title="Phase Peak #2",
        xlabel="Frequency (Hz)",
        ylabel="Phase (°)",
        xscale="log",
        legend=temp,
    )

    fig.suptitle(path.name, weight="bold", size="x-large")
    fig.tight_layout(h_pad=3, w_pad=3, rect=(0, 0, 1, 0.95))
    fig.savefig(path.as_posix(), dpi=300)
    plt.close()


def _to_polar(xy_data):
    z_data = xy_data[1] + 1j * xy_data[2]

    polar_data = np.empty_like(xy_data)
    polar_data[0] = xy_data[0]
    polar_data[1] = np.abs(z_data)
    polar_data[2] = np.angle(z_data, deg=True)

    return polar_data


def _draw_ax(axes, freq, data=None, fit=None, **kwargs):
    """Draws plots using the matplotlib.axes.Axes class.

    Args:
        axes: The "Axes" object where the plots are drawn.
        freq: An array or list of arrays of the x values.
        data: An array or list of arrays of y values plotted with
            markers.
        fit: An array or list of arrays of y values plotted with lines.
        kwargs: Optional keyword arguments:
            - title: The title string.
            - xlabel: The xlabel strings.
            - ylabel: The ylabel strings.
            - xscale: ['linear' | 'log' | 'logit' | 'symlog']
            - legend: An iterable of strings for the legend.

    """
    if not isinstance(freq, list):
        freq = [freq]

    if fit is not None:
        if not isinstance(fit, list):
            fit = [fit]
        for i, j in zip(freq, fit):
            axes.plot(i, j)

    if data is not None:
        if not isinstance(data, list):
            data = [data]
        for i, j in zip(freq, data):
            axes.plot(i, j, marker=".")

    if "legend" in kwargs:
        axes.legend(kwargs["legend"])

    axes.set_xlabel(kwargs.get("xlabel", ""))
    axes.set_ylabel(kwargs.get("ylabel", ""))

    axes.set_xscale(kwargs.get("xscale", "linear"))
    axes.set_xticklabels(["{:g}".format(i) for i in axes.get_xticks()])

    axes.set_title(kwargs.get("title", ""))
    axes.grid(True)


def _fit_asym2sig(position, voltage):
    """Return a asymmetric double sigmoidal fit and its parameters.

    Args:
        position: An array of floats of the position.
        voltage: An array of floats of the voltage.

    """
    pos_max = position[voltage.argmax()]
    pos_min = position[voltage.argmin()]
    baseline = (max(voltage) + min(voltage)) / 2

    init_params = [
        baseline,
        max(voltage) - baseline,
        min(voltage) - baseline,
        pos_max - (pos_min - pos_max) / 4,
        pos_max + (pos_min - pos_max) / 4,
        pos_min - (pos_min - pos_max) / 4,
        pos_min + (pos_min - pos_max) / 4,
        2,
        2,
        2,
        2,
    ]

    residuals = lambda pfit: voltage - _asym2sig(pfit, position)

    pfit = least_squares(residuals, init_params).x
    fit = _asym2sig(pfit, position)

    return fit, pfit[:3]


def _asym2sig(params, pos):
    """Return an asymmetric double sigmoidal function of the position.

    The asymmetric double sigmoidal function is the difference of two
    independent sigmoidal functions. They are expressed here as 
    hyperbolic tangents in order to reduce the computer time.

    Args:
        params: A list of all the function parameters.
        pos: A float or array of floats of the position.

    """
    u_0, u_max, u_min, xc_1, xc_2, xc_3, xc_4, w_1, w_2, w_3, w_4 = params
    return (
        u_0
        + u_max * (np.tanh((pos - xc_1) / w_1) - np.tanh((pos - xc_2) / w_2)) / 2
        + u_min * (np.tanh((pos - xc_3) / w_3) - np.tanh((pos - xc_4) / w_4)) / 2
    )
