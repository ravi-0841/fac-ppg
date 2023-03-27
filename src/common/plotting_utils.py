# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Modified from https://github.com/NVIDIA/tacotron2"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import pylab


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram + 1e-10, aspect="auto", origin="lower",
                   interpolation='none')
    # plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    return fig


def plot_posterior_to_numpy(posterior):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.plot(posterior.reshape(-1,))
    plt.xlabel("Frames")
    plt.ylabel("Mask Value")
    plt.tight_layout()

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    return fig


def plot_1d_signal_numpy(signal):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.plot(signal.reshape(-1,))
    plt.xlabel("Time/Frames")
    plt.ylabel("Signal Value")
    plt.tight_layout()

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    return fig


def plot_ppg_to_numpy(ppg):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(ppg, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Phonemes")
    plt.ylabel("Time")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def plot_saliency_to_numpy(saliency):
    fig, ax = plt.subplots(figsize=(12, 3))
    # im = ax.bar(np.arange(len(saliency)), saliency)
    im = ax.bar(["neu", "ang", "hap", "sad", "fea"], saliency, color="red", alpha=0.7)
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.tight_layout()

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    return fig


def plot_rate_to_numpy(rate, rate_classes):
    fig, ax = plt.subplots(figsize=(12, 3))
    # im = ax.bar(np.arange(len(saliency)), saliency)
    if rate_classes is not None:
        im = ax.bar(rate_classes, rate, alpha=0.7)
    elif len(rate) == 7:
        im = ax.bar(["0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"], rate)
    elif len(rate) == 11:
        im = ax.bar(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5"], rate)
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.tight_layout()

    fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    return fig
