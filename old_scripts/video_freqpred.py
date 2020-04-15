import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from helper_functions import interpolate_data
from helper_functions import print_progress
from helper_functions import update_viewport

# some globals
# either this
LAYOUT = "3x2"
RES_WIDTH = 1920
RES_HEIGHT = 1080

# or this
# LAYOUT = "6x2"
# RES_WIDTH = 1080
# RES_HEIGHT = 1920

NUM_INTERPOLATION = 5
DPI = 100
TOP_MARIGIN_PX = 40
NEURON_0_LABEL = "Neuron 0"
NEURON_1_LABEL = "Neuron 1"
NEURON_0_COLOR = "#fe6100"  # orange
NEURON_1_COLOR = "#321c4c"  # dark violet
NEURONS_LINE_STYLE = "-"
NEURONS_LINE_WIDTH = 2
GT_0_LABEL = "Ground Truth 0"
GT_1_LABEL = "Ground Truth 1"
GT_0_COLOR = "#464646"  # gray
GT_1_COLOR = "k"
GT_LINE_STYLE = "-"
GT_LINE_WIDTH = 2
DYNAMIC_VIEW = True
USE_INTERPOLATION = False
ARTIFICIAL_FRAME_MULTIPLIER = 5
VIEWPORT_WIDTH = 100
global LATEST_DP
LATEST_DP = 0

# big fonts
XLABEL_FS = 13
YLABEL_FS = 13
SUBPLOT_TITLE_FS = 15
FIG_FS = 18
TICKS_FS = 13

# load the data
plstm = np.transpose(np.load("eval/freqpred/vctrnn_freqpred_lstm_d1803271405_predvals.npz")['arr_0'][0][:][:])
pactrnn = np.transpose(np.load("eval/freqpred/vctrnn_freqpred_mactrnn_d1803271237_predvals.npz")['arr_0'][0][:][:])
pmavctrnn = np.transpose(
    np.load('eval/freqpred/vctrnn_freqpred_mavctrnn_d1803271411_predvals.npz')['arr_0'][0][:][:])
pmctrnn = np.transpose(np.load("eval/freqpred/vctrnn_freqpred_mctrnn_d1803271447_predvals.npz")['arr_0'][0][:][:])
pmvctrnn = np.transpose(np.load("eval/freqpred/vctrnn_freqpred_mvctrnn_d1803271321_predvals.npz")['arr_0'][0][:][:])
psrn = np.transpose(np.load('eval/freqpred/vctrnn_freqpred_srn_d1803271449_predvals.npz')['arr_0'][0][:][:])
target = np.transpose(np.load('eval/freqpred/vctrnn_freqpred_test_targetvals.npz')['arr_0'][0, :, :])

# interpolate everything
if USE_INTERPOLATION:
    extended_lstm = interpolate_data(NUM_INTERPOLATION, plstm)
    extended_actrnn = interpolate_data(NUM_INTERPOLATION, pactrnn)
    extended_mavctrnn = interpolate_data(NUM_INTERPOLATION, pmavctrnn)
    extended_mctrnn = interpolate_data(NUM_INTERPOLATION, pmctrnn)
    extended_mvctrnn = interpolate_data(NUM_INTERPOLATION, pmvctrnn)
    extended_srn = interpolate_data(NUM_INTERPOLATION, psrn)
    extended_target = interpolate_data(NUM_INTERPOLATION, target)
else:
    extended_lstm = plstm
    extended_actrnn = pactrnn
    extended_mavctrnn = pmavctrnn
    extended_mctrnn = pmctrnn
    extended_mvctrnn = pmvctrnn
    extended_srn = psrn
    extended_target = target

# setup figure
fig = plt.figure(figsize=(RES_WIDTH / DPI, RES_HEIGHT / DPI))
if LAYOUT == "3x2":
    ax_srn = fig.add_subplot(321)
    ax_actrnn = fig.add_subplot(322)
    ax_ctrnn = fig.add_subplot(323)
    ax_vctrnn = fig.add_subplot(324)
    ax_lstm = fig.add_subplot(325)
    ax_avctrnn = fig.add_subplot(326)

    # for later iteration
    net_axes = [ax_srn, ax_actrnn, ax_ctrnn, ax_vctrnn, ax_lstm, ax_avctrnn]

    # set axes titles
    ax_srn.set_title("SRN", fontsize=SUBPLOT_TITLE_FS)
    ax_actrnn.set_title("ACTRNN", fontsize=SUBPLOT_TITLE_FS)
    ax_ctrnn.set_title("CTRNN", fontsize=SUBPLOT_TITLE_FS)
    ax_vctrnn.set_title("VCTRNN", fontsize=SUBPLOT_TITLE_FS)
    ax_lstm.set_title("LSTM", fontsize=SUBPLOT_TITLE_FS)
    ax_avctrnn.set_title("AVCTRNN", fontsize=SUBPLOT_TITLE_FS)

    # set y-axes labels
    ax_srn.set_ylabel("Act. SRN", fontsize=YLABEL_FS)
    ax_actrnn.set_ylabel("Act. ACTRNN", fontsize=YLABEL_FS)
    ax_ctrnn.set_ylabel("Act. CTRNN", fontsize=YLABEL_FS)
    ax_vctrnn.set_ylabel("Act. VCTRNN", fontsize=YLABEL_FS)
    ax_lstm.set_ylabel("Act. LSTM", fontsize=YLABEL_FS)
    ax_avctrnn.set_ylabel("Act. AVCTRNN", fontsize=YLABEL_FS)

    colors = sns.color_palette("Set2")

    # set x-lim, x-lim, x-label, background colors (everything that is the same for all axes)
    for axis in net_axes:
        axis.set_xlim(0, len(target[0]))
        axis.set_ylim(-1, 1)
        axis.set_xlabel("Timestep", fontsize=XLABEL_FS)
        axis.axvspan(0, 200, facecolor=colors[-1], alpha=0.5)
        axis.axvspan(200, 450, facecolor="#fbeaae", alpha=0.5)

    # init line artists
    srn_n0, = ax_srn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=NEURON_0_COLOR)
    srn_n1, = ax_srn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=NEURON_1_COLOR)
    actrnn_n0 = ax_actrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                               c=NEURON_0_COLOR)
    actrnn_n1 = ax_actrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                               c=NEURON_1_COLOR)
    ctrnn_n0 = ax_ctrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                             c=NEURON_0_COLOR)
    ctrnn_n1 = ax_ctrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                             c=NEURON_1_COLOR)
    vctrnn_n0 = ax_vctrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                               c=NEURON_0_COLOR)
    vctrnn_n1 = ax_vctrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                               c=NEURON_1_COLOR)
    lstm_n0 = ax_lstm.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=NEURON_0_COLOR)
    lstm_n1 = ax_lstm.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=NEURON_1_COLOR)
    avctrnn_n0, = ax_avctrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                                  c=NEURON_0_COLOR)
    avctrnn_n1, = ax_avctrnn.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                                  c=NEURON_1_COLOR)

    # prepare list of artists for later iteration
    net_artists = [[srn_n0, srn_n1], [actrnn_n0, actrnn_n1], [ctrnn_n0, ctrnn_n1], [vctrnn_n0, vctrnn_n1],
                   [lstm_n0, lstm_n1], [avctrnn_n0, avctrnn_n1]]

    # same with data...
    net_data = [extended_srn, extended_actrnn, extended_mctrnn, extended_mvctrnn, extended_lstm, extended_mavctrnn]

    # plot static ground truth activations
    gt_x = np.linspace(0, np.shape(psrn)[1], np.shape(extended_target)[1])
    gt_y0 = extended_target[0]
    gt_y1 = extended_target[1]
    for axis in net_axes:
        axis.plot(gt_x, gt_y0, c=GT_0_COLOR, lw=GT_LINE_WIDTH, label=GT_0_LABEL, ls=GT_LINE_STYLE)
        axis.plot(gt_x, gt_y1, c=GT_1_COLOR, lw=GT_LINE_WIDTH, label=GT_1_LABEL, ls=GT_LINE_STYLE)

elif LAYOUT == "6x2":
    ax_srn_0 = fig.add_subplot(621)
    ax_srn_1 = fig.add_subplot(622)
    ax_actrnn_0 = fig.add_subplot(623)
    ax_actrnn_1 = fig.add_subplot(624)
    ax_ctrnn_0 = fig.add_subplot(625)
    ax_ctrnn_1 = fig.add_subplot(626)
    ax_vctrnn_0 = fig.add_subplot(627)
    ax_vctrnn_1 = fig.add_subplot(628)
    ax_lstm_0 = fig.add_subplot(629)
    ax_lstm_1 = fig.add_subplot(6, 2, 10)
    ax_avctrnn_0 = fig.add_subplot(6, 2, 11)
    ax_avctrnn_1 = fig.add_subplot(6, 2, 12)

    # for later iteration
    net_axes = [ax_srn_0, ax_srn_1, ax_actrnn_0, ax_actrnn_1, ax_ctrnn_0, ax_ctrnn_1, ax_vctrnn_0, ax_vctrnn_1,
                ax_lstm_0, ax_lstm_1, ax_avctrnn_0, ax_avctrnn_1]

    # set axes titles
    ax_srn_0.set_title("SRN Neuron 0")
    ax_srn_1.set_title("SRN Neuron 1")
    ax_actrnn_0.set_title("ACTRNN Neuron 0")
    ax_actrnn_1.set_title("ACTRNN Neuron 1")
    ax_ctrnn_0.set_title("CTRNN Neuron 0")
    ax_ctrnn_0.set_title("CTRNN Neuron 1")
    ax_vctrnn_0.set_title("VCTRNN Neuron 0")
    ax_vctrnn_1.set_title("VCTRNN Neuron 1")
    ax_lstm_0.set_title("LSTM Neuron 0")
    ax_lstm_1.set_title("LSTM Neuron 1")
    ax_avctrnn_0.set_title("AVCTRNN Neuron 0")
    ax_avctrnn_0.set_title("AVCTRNN Neuron 1")

    # set y-axes labels
    ax_srn_0.set_ylabel("Act. SRN Neuron 0")
    ax_srn_1.set_ylabel("Act. SRN Neuron 1")
    ax_actrnn_0.set_ylabel("Act. ACTRNN Neuron 0")
    ax_actrnn_1.set_ylabel("Act. ACTRNN Neuron 1")
    ax_ctrnn_0.set_ylabel("Act. CTRNN Neuron 0")
    ax_ctrnn_0.set_ylabel("Act. CTRNN Neuron 1")
    ax_vctrnn_0.set_ylabel("Act. VCTRNN Neuron 0")
    ax_vctrnn_1.set_ylabel("Act. VCTRNN Neuron 1")
    ax_lstm_0.set_ylabel("Act. LSTM Neuron 0")
    ax_lstm_1.set_ylabel("Act. LSTM Neuron 1")
    ax_avctrnn_0.set_ylabel("Act. AVCTRNN Neuron 0")
    ax_avctrnn_0.set_ylabel("Act. AVCTRNN Neuron 1")

    # set x-lim, x-lim, x-label, background colors (everything that is the same for all axes)
    for axis in net_axes:
        axis.set_xlim(0, len(target[0]))
        axis.set_ylim(-1, 1)
        axis.set_xlabel("Timestep")
        axis.axvspan(0, 200, facecolor='gray', alpha=0.5)
        axis.axvspan(200, 399, facecolor='yellow', alpha=0.5)

    # init line artists
    srn_n0, = ax_srn_0.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                            c=NEURON_0_COLOR)
    srn_n1, = ax_srn_1.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                            c=NEURON_1_COLOR)
    actrnn_n0 = ax_actrnn_0.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                                 c=NEURON_0_COLOR)
    actrnn_n1 = ax_actrnn_1.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                                 c=NEURON_1_COLOR)
    ctrnn_n0 = ax_ctrnn_0.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                               c=NEURON_0_COLOR)
    ctrnn_n1 = ax_ctrnn_1.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                               c=NEURON_1_COLOR)
    vctrnn_n0 = ax_vctrnn_0.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                                 c=NEURON_0_COLOR)
    vctrnn_n1 = ax_vctrnn_1.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                                 c=NEURON_1_COLOR)
    lstm_n0 = ax_lstm_0.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                             c=NEURON_0_COLOR)
    lstm_n1 = ax_lstm_1.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                             c=NEURON_1_COLOR)
    avctrnn_n0, = ax_avctrnn_0.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE,
                                    c=NEURON_0_COLOR)
    avctrnn_n1, = ax_avctrnn_1.plot([], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE,
                                    c=NEURON_1_COLOR)

    # prepare list of artists for later iteration
    net_artists = [srn_n0, srn_n1, actrnn_n0, actrnn_n1, ctrnn_n0, ctrnn_n1, vctrnn_n0, vctrnn_n1, lstm_n0, lstm_n1,
                   avctrnn_n0, avctrnn_n1]

    # same with data...
    net_data = [extended_srn[0], extended_srn[1], extended_actrnn[0], extended_actrnn[1], extended_mctrnn[0],
                extended_mctrnn[1], extended_mvctrnn[0], extended_mvctrnn[1], extended_lstm[0], extended_lstm[1],
                extended_mavctrnn[0], extended_mavctrnn[1]]

    # plot static ground truth activations
    gt_x = np.linspace(0, np.shape(psrn)[1], np.shape(extended_target)[1])
    gt_y0 = extended_target[0]
    gt_y1 = extended_target[1]
    for index, axis in enumerate(net_axes):
        if index % 2 == 0:
            axis.plot(gt_x, gt_y0, c=GT_0_COLOR, lw=GT_LINE_WIDTH, label=GT_0_LABEL)
        elif index % 2 == 1:
            axis.plot(gt_x, gt_y1, c=GT_1_COLOR, lw=GT_LINE_WIDTH, label=GT_1_LABEL)

# set title
fig.suptitle("Human Motion Patterns Prediction", fontfamily="sans-serif", fontweight="bold", fontsize=FIG_FS)

# restrict the area of the figures to leave TOP_MARGIN pixel free for title
top_percentage = TOP_MARIGIN_PX / RES_HEIGHT
plt.tight_layout(rect=(0, 0, 1, 1 - top_percentage))


def animate(i):
    global LATEST_DP
    gt_x = np.linspace(0, np.shape(psrn)[1], np.shape(extended_target)[1])
    flat_artists = []

    if ARTIFICIAL_FRAME_MULTIPLIER and not USE_INTERPOLATION:
        print_progress(i, len(extended_target[0]) * ARTIFICIAL_FRAME_MULTIPLIER, prefix="Progress", suffix="Complete")
    else:
        print_progress(i, len(extended_target[0]), prefix="Progress", suffix="Complete")

    if LAYOUT == "3x2":
        if ARTIFICIAL_FRAME_MULTIPLIER and not USE_INTERPOLATION:
            if i % ARTIFICIAL_FRAME_MULTIPLIER == 0 and i > 0:  # here we only add a new datapoint every x frames
                LATEST_DP += 1
                x = gt_x[0:LATEST_DP]

                for artists, data, axis in zip(net_artists, net_data, net_axes):
                    for neuron in range(len(artists)):
                        # append new y
                        activations = data[neuron][0:LATEST_DP]

                        # set new data on the artist
                        try:
                            artists[neuron].set_data(x, activations)
                            flat_artists.append(artists[neuron])
                        except AttributeError:
                            # this occurs, because at the artist initialization above, mpl returns either the artists directly, or a list containing the artists...
                            # i am baffled as to why thiat is or whether that is intended, either way this is should do the trick...
                            artists[neuron][0].set_data(x, activations)
                            flat_artists.append(artists[neuron][0])

            # update the x-lim on every axes so that we "slide" across the graph
            update_viewport(mode="art_frames", frame=i, viewport_width=VIEWPORT_WIDTH, x_ticks=gt_x, axes=net_axes, latest_dp=LATEST_DP, art_frame_mult=ARTIFICIAL_FRAME_MULTIPLIER)

        else:
            x = gt_x[0:i]
            for artists, data, axis in zip(net_artists, net_data, net_axes):
                for neuron in range(len(artists)):
                    # append new y
                    activations = data[neuron][0:i]

                    # set new data on the artist
                    try:
                        artists[neuron].set_data(x, activations)
                        flat_artists.append(artists[neuron])
                    except AttributeError:
                        # this occurs, because at the artist initialization above, mpl returns either the artists directly, or a list containing the artists...
                        # i am baffled why this is other whether that is intended, either way this is should do the trick...
                        artists[neuron][0].set_data(x, activations)
                        flat_artists.append(artists[neuron][0])

                # update the x-lim on every axes so that we "slide" across the graph
                update_viewport(mode="inerpolation", frame=i, viewport_width=VIEWPORT_WIDTH, x_ticks=gt_x, axes=net_axes)

    elif LAYOUT == "6x2":
        # TODO use proper x-lim updates here as well
        for artist, data, axis in zip(net_artists, net_data, net_axes):
            # append new y
            activations = data[0:i]

            # set new data on the artist
            try:
                artist.set_data(x, activations)
                flat_artists.append(artist)
            except AttributeError:
                # this occurs, because at the artist initialization above, mpl returns either the artists directly, or a list containing the artists...
                # i am baffled why this is other whether that is intended, either way this is should do the trick...
                artist[0].set_data(x, activations)
                flat_artists.append(artist[0])

            # update the xlim on both axes so that we "slide" across the graph
            if DYNAMIC_VIEW:
                x_min = gt_x[i] - 50
                if x_min < 0:
                    x_min = 0

                x_max = gt_x[i] + 50
                if x_max < 100:
                    x_max = 100

                # update the viewport
                axis.set_xlim(x_min, x_max)
                axis.figure.canvas.draw()

    return flat_artists  # return sequence of artists


if not USE_INTERPOLATION and ARTIFICIAL_FRAME_MULTIPLIER:
    ani = animation.FuncAnimation(fig, animate, frames=len(extended_target[0] / 2) * ARTIFICIAL_FRAME_MULTIPLIER)
    ani.save(filename='animated_freqpreds.mp4', dpi=100, fps=90)
else:
    ani = animation.FuncAnimation(fig, animate, frames=len(extended_target[0]), blit=True)
    ani.save(filename='animated_freqpreds.mp4', dpi=100, fps=90)

