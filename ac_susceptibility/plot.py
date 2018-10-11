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
import matplotlib.pyplot as plt

from .load import load, sorted_subfolders, list_freqs_and_files
from .xyfit import xyfit


def plot(data_path, skip_voltage, calibration_data):
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

                data = load(voltage_file)
                fit, pfit = xyfit(data, calibration_data)
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


def make_voltage_plot(data, fit, path):
    """Plot amplitude and phase of the signal versus sample position.

    Args:
        data: An array containing the position of the sample, the
            amplitude of the voltage measured by the lock-in amplifier,
            and its phase.
        fit: Same object as "data" with fitted values.
        path: The path of the saved plot as a "Path" object.

    """
    print(f'Saving plot "{path}"...')

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
    print(f'Saving plot "{path.name}"...')

    if not isinstance(data[0], list):
        print("data is not a list")
        data = [data]

    temp = []
    curves = [[], [], [], [], [], [], []]

    for i, j in data:
        temp.append(i)
        curves[0].append(j[:, 0])
        curves[1].append(j[:, 1] / j[:, 0])
        curves[2].append(j[:, 2] / j[:, 0])
        curves[3].append(j[:, 3] / j[:, 0])
        # curves[2].append(j[:, 2] / j[:, 1])
        # curves[3].append(j[:, 3] / j[:, 1])
        curves[4].append(j[:, 4])
        curves[5].append(j[:, 5])
        curves[6].append(j[:, 6])

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
    axes.set_xticklabels([f"{i:g}" for i in axes.get_xticks()])

    axes.set_title(kwargs.get("title", ""))
    axes.grid(True)
