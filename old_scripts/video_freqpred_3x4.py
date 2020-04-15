import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helper_functions import update_viewport
from helper_functions import print_progress
import seaborn as sns

# globals
RES_WIDTH = 1920
RES_HEIGHT = 1080
DPI = 100
NEURON_0_LABEL = "Neuron 0"
NEURON_1_LABEL = "Neuron 1"
NEURONS_LINE_STYLE = "-"
NEURONS_LINE_WIDTH = 1
GT_0_LABEL = "Ground Truth 0"
GT_1_LABEL = "Ground Truth 1"
GT_LINE_STYLE = ":"
GT_LINE_WIDTH = 1
ARTIFICIAL_FRAME_MULTIPLIER = 2
FPS = 30
LATEST_DP = 0
VIEWPORT_WIDTH = 100
FRAMES = None
# COLORS = [(0.0, 0.0, 0.7), (1.0, 0.3, 0.3), (1.0, 0.0, 1.0), (0.0, 0.7, 0.7), (0.7, 0.7, 0.7), (0.5, 0.5, 0.7)]
COLORS = sns.color_palette("muted")

# big fonts:
TOP_MARIGIN_PX = 50
XLABEL_FS = 13
YLABEL_FS = 13
SUBPLOT_TITLE_FS = 15
FIG_FS = 18
TICKS_FS = 13


# load the data for the plots
def load_data():
    # model name and corresponding batch id
    model_names = [["mctrnn", "_d1803271447"], ["mactrnn", "_d1803271237"], ["mvctrnn", "_d1803271321"],
                   ["mavctrnn", "_d1803271411"], ["lstm", "_d1803271405"], ["srn", "_d1803271449"]]

    # load data for every model
    model_data = []
    tau_data = {}
    sig_data = {}
    act_data = {}
    pred_data = {}
    for name_id in model_names:
        with np.load("../eval/freqpred/vctrnn_freqpred_" + name_id[0] + name_id[1] + "_tausigmavals.npz") as f:
            tau_sigma = f['arr_0']

            # srn and lstm dont have tau and sigma data...
            if name_id[0] != 'lstm' and name_id[0] != 'srn':
                sigma = tau_sigma[-1, 1, :]
                tau = tau_sigma[-1, 0, :]

                tau_data[name_id[0]] = tau
                sig_data[name_id[0]] = sigma

        if name_id[0] == "lstm":
            name = "mlstm"
        elif name_id[0] == "srn":
            name = "msrn"
        else:
            name = name_id[0]

        load_model = {
            "name": name,
            "preds": np.transpose(
                np.load("../eval/freqpred/vctrnn_freqpred_"
                        + name_id[0] + name_id[1]
                        + "_predvals.npz")['arr_0'][0][:][:]),
            "hid_acts": np.transpose(
                np.load("../eval/freqpred/vctrnn_freqpred_"
                        + name_id[0] + name_id[1]
                        + "_hidavals.npz")['arr_0'][0][:][:]),
            "taus": tau,
            "sigmas": sigma
        }
        model_data.append(load_model)

        act_data[name] = load_model["hid_acts"]
        pred_data[name] = load_model["preds"]

    # pred_data = [model["preds"] for model in model_data]

    return model_data, pred_data, act_data, tau_data, sig_data


# add every axes to the figure
def add_axes(fig):
    ax_ctrnn_acts_tau = fig.add_subplot(3, 4, 1)
    ax_actrnn_acts_tau = fig.add_subplot(3, 4, 2)
    ax_vctrnn_acts_tau = fig.add_subplot(3, 4, 4)
    ax_avctrnn_acts_tau = fig.add_subplot(3, 4, 3)

    ax_ctrnn_preds = fig.add_subplot(3, 4, 5)
    ax_actrnn_preds = fig.add_subplot(3, 4, 6)
    ax_vctrnn_preds = fig.add_subplot(3, 4, 8)
    ax_avctrnn_preds = fig.add_subplot(3, 4, 7)

    # ax_ctrnn_acts_sig = fig.add_subplot(3, 4, 9)
    # ax_actrnn_acts_sig = fig.add_subplot(3, 4, 10)
    ax_lstm_preds = fig.add_subplot(3, 4, 9)
    ax_srn_preds = fig.add_subplot(3, 4, 10)
    ax_vctrnn_acts_sig = fig.add_subplot(3, 4, 11)
    ax_avctrnn_axts_sig = fig.add_subplot(3, 4, 12)

    # for later iteration
    # axes = [ax_ctrnn_preds, ax_actrnn_preds, ax_vctrnn_preds, ax_avctrnn_preds, ax_ctrnn_acts_tau, ax_actrnn_acts_tau,
    #         ax_vctrnn_acts_tau, ax_avctrnn_acts_tau, ax_ctrnn_acts_sig, ax_actrnn_acts_sig, ax_vctrnn_acts_sig,
    #         ax_avctrnn_axts_sig]
    axes = [ax_ctrnn_preds, ax_actrnn_preds, ax_vctrnn_preds, ax_avctrnn_preds, ax_ctrnn_acts_tau, ax_actrnn_acts_tau,
            ax_vctrnn_acts_tau, ax_avctrnn_acts_tau, ax_lstm_preds, ax_srn_preds, ax_vctrnn_acts_sig,
            ax_avctrnn_axts_sig]

    pred_axes = {"mctrnn": ax_ctrnn_preds,
                 "mactrnn": ax_actrnn_preds,
                 "mvctrnn": ax_vctrnn_preds,
                 "mavctrnn": ax_avctrnn_preds,
                 "mlstm": ax_lstm_preds,
                 "msrn": ax_srn_preds}

    act_tau_axes = {"mctrnn": ax_ctrnn_acts_tau,
                    "mactrnn": ax_actrnn_acts_tau,
                    "mvctrnn": ax_vctrnn_acts_tau,
                    "mavctrnn": ax_avctrnn_acts_tau}

    # act_sig_axes = {"mctrnn": ax_ctrnn_acts_sig,
    #                 "mactrnn": ax_actrnn_acts_sig,
    #                 "mvctrnn": ax_vctrnn_acts_sig,
    #                 "mavctrnn": ax_avctrnn_axts_sig}
    act_sig_axes = {"mvctrnn": ax_vctrnn_acts_sig,
                    "mavctrnn": ax_avctrnn_axts_sig}

    return axes, pred_axes, act_tau_axes, act_sig_axes


def main():
    # load data
    models, pred_data, act_data, tau_data, sig_data = load_data()

    # setup figure
    fig = plt.figure(figsize=(RES_WIDTH / DPI, RES_HEIGHT / DPI))
    fig.suptitle("A nice title", fontfamily="sans-serif", fontweight="bold", fontsize=FIG_FS)

    # add axes for each plot
    axes, pred_axes, act_tau_axes, act_sig_axes = add_axes(fig)

    # set axis titles
    for model in models:
        # set titles on predictions plots
        for key in pred_axes.keys():
            if model["name"] == key:
                model_name = model["name"][1:]  # remove the trailing "m"
                pred_axes[key].set_title(model_name.upper() + " predictions", fontsize=SUBPLOT_TITLE_FS)
                pred_axes[key].set_xlabel("Timestep", fontsize=XLABEL_FS)
                pred_axes[key].set_ylabel("Hidden actiation", fontsize=YLABEL_FS)

        # set titles on activation over tau plots
        for key in act_tau_axes.keys():
            if model["name"] == key:
                model_name = model["name"][1:]  # remove the trailing "m"
                act_tau_axes[key].set_title(
                    model_name.upper() + " activation over timescale", fontsize=SUBPLOT_TITLE_FS)
                act_tau_axes[key].set_xlabel(r"Timescale $\tau$", fontsize=XLABEL_FS)
                act_tau_axes[key].set_ylabel("Hidden activation", fontsize=YLABEL_FS)

        # set titles on activation over sigma plots
        for key in act_sig_axes.keys():
            if model["name"] == key:
                model_name = model["name"][1:]  # remove the trailing "m"
                act_sig_axes[key].set_title(model_name.upper() + " activation over variance", fontsize=SUBPLOT_TITLE_FS)
                act_sig_axes[key].set_xlabel(r"Variance $\sigma$", fontsize=XLABEL_FS)
                act_sig_axes[key].set_ylabel("Hidden activation", fontsize=YLABEL_FS)

    # set y-lim on every axis
    for axis in axes:
        axis.set_ylim(-1, 1)

    # init line artists
    # row 1: prediction plots
    ctrnn_pred_n0 = pred_axes["mctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[0])
    ctrnn_pred_n1 = pred_axes["mctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[1])

    actrnn_pred_n0 = pred_axes["mactrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[0])
    actrnn_pred_n1 = pred_axes["mactrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[1])

    vctrnn_pred_0 = pred_axes["mvctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[0])
    vctrnn_pred_1 = pred_axes["mvctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[1])

    avctrnn_pred_0 = pred_axes["mavctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[0])
    avctrnn_pred_1 = pred_axes["mavctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[1])

    lstm_pred_0 = pred_axes["mlstm"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[0])
    lstm_pred_1 = pred_axes["mlstm"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[1])

    srn_pred_0 = pred_axes["msrn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[0])
    srn_pred_1 = pred_axes["msrn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=COLORS[1])

    # for later iteration
    pred_artists = [
        [ctrnn_pred_n0, ctrnn_pred_n1], [actrnn_pred_n0, actrnn_pred_n1], [vctrnn_pred_0, vctrnn_pred_1],
        [avctrnn_pred_0, avctrnn_pred_1], [lstm_pred_0, lstm_pred_1], [srn_pred_0, srn_pred_1]]

    # load target preds outside of loops
    target_preds = np.transpose(np.load('../code/data/vctrnn_freqpred_test_targetvals.npz')['arr_0'][0, :, :])

    # draw gt prediction for pred plots
    gt_x = np.arange(0, np.shape(target_preds)[1])
    gt_y0 = target_preds[0]
    gt_y1 = target_preds[1]
    for ax_pred_plot in pred_axes.values():
        ax_pred_plot.plot(gt_x, gt_y0, c=COLORS[0], alpha=0.3, lw=GT_LINE_WIDTH, label=GT_0_LABEL)
        ax_pred_plot.plot(gt_x, gt_y1, c=COLORS[1], alpha=0.3, lw=GT_LINE_WIDTH, label=GT_1_LABEL)

        ax_pred_plot.axvspan(0, 200, facecolor=sns.color_palette("muted")[7], alpha=0.3)  # muted [7]
        ax_pred_plot.axvspan(200, 450, facecolor=sns.color_palette("muted")[8], alpha=0.3)

    # row 2: hidden activations over tau
    ctrnn_act_tau = act_tau_axes["mctrnn"].scatter([], [], c=COLORS[4])

    actrnn_act_tau = act_tau_axes["mactrnn"].scatter([], [], c=COLORS[4])
    act_tau_axes["mactrnn"].set_xticklabels([123.45])

    vctrnn_act_tau = act_tau_axes["mvctrnn"].scatter([], [], c=COLORS[4])

    avctrnn_act_tau = act_tau_axes["mavctrnn"].scatter([], [], c=COLORS[4])
    act_tau_axes["mavctrnn"].set_xticklabels([123.45])

    act_tau_artists = {
        "mctrnn": ctrnn_act_tau,
        "mactrnn": actrnn_act_tau,
        "mvctrnn": vctrnn_act_tau,
        "mavctrnn": avctrnn_act_tau
    }

    # add horizontal bar on 0 to all activation plots
    for axis in act_tau_axes.values():
        axis.axhline(c="k", zorder=-1, lw=1)

    # setup xmin xmax and x_ticks
    for model_tau in tau_data:

        # find unique tau vals
        distinct_tau = []
        for tau in tau_data[model_tau]:
            if tau not in distinct_tau:
                distinct_tau.append(tau)

        distinct_tau.sort()

        # create dict mapping from tau vals to tau x indices
        tau_x_vals = {}
        tau_x_ind = 0
        for dist_tau in distinct_tau:
            tau_x_vals[dist_tau] = tau_x_ind
            tau_x_ind += 1

        # set x ticks
        x_ticks = list(tau_x_vals.values())
        x_ticks.sort()
        act_tau_axes[model_tau].set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
        act_tau_axes[model_tau].set_xticks(x_ticks)

        # set x ticks labels
        x_tick_labels = [str(np.around(tau, decimals=2)) for tau in distinct_tau]
        # if we have many x vals, only display four values: first, last and two in-between
        if len(x_tick_labels) > 5:
            for j in range(len(x_tick_labels)):
                # here, we only display four x_tick_lables, thus we set empty strings as every other tick label
                inds = [int(np.around(ind)) for ind in np.linspace(0, len(x_tick_labels) - 1, 4)]
                if j in inds:
                    pass
                else:
                    x_tick_labels[j] = ""

            act_tau_axes[model_tau].set_xticklabels(x_tick_labels)

        else:
            act_tau_axes[model_tau].set_xticklabels(x_tick_labels)

    # row 3: hidden activations over sigma for vctrnn and avctrnn (lstm and srn preds have been added earlier)
    vctrnn_act_sig = act_sig_axes["mvctrnn"].scatter([], [], c=COLORS[3])
    avctrnn_act_sig = act_sig_axes["mavctrnn"].scatter([], [], c=COLORS[3])

    act_sig_artists = {
        "mvctrnn": vctrnn_act_sig,
        "mavctrnn": avctrnn_act_sig
    }

    # add horizontal bar on 0 to all activation plots
    for axis in act_sig_axes.values():
        axis.axhline(c="k", zorder=-1, lw=1)

    # setup xmin xmax and x_ticks
    for model_sig in sig_data:

        # we only want to plot lstm and srn prediction plot, no activation over sigma
        if model_sig == "mlstm" or model_sig == "msrn":
            continue

        #  we no longer want to plot this, instead we use the space to plot the lstm and srn activations
        if model_sig == "mctrnn" or model_sig == "mactrnn":
            continue

        artists = act_sig_artists[model_sig]
        model_xy = []
        # find unique sigma vals
        distinct_sig = []
        for sig in sig_data[model_sig]:
            if sig not in distinct_sig:
                distinct_sig.append(sig)

        distinct_sig.sort()

        # create dict mapping from sigma vals to sigma x indices
        sig_x_vals = {}
        sig_x_ind = 0
        for dist_tau in distinct_sig:
            sig_x_vals[dist_tau] = sig_x_ind
            sig_x_ind += 1

        # set xticks
        x_ticks = list(sig_x_vals.values())
        x_ticks.sort()
        act_sig_axes[model_sig].set_xticks(x_ticks)
        act_sig_axes[model_sig].set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)

        # set xticks labels
        x_tick_labels = [str(np.around(sig, decimals=2)) for sig in distinct_sig]
        # if we have many x vals, only display four values: first, last and two in-between
        if len(x_tick_labels) > 5:
            for j in range(len(x_tick_labels)):
                # here, we only display four x_tick_lables, thus we set empty strings as every other tick label
                inds = [int(np.around(ind)) for ind in np.linspace(0, len(x_tick_labels) - 1, 4)]
                if j in inds:
                    pass
                else:
                    x_tick_labels[j] = ""

            act_sig_axes[model_sig].set_xticklabels(x_tick_labels)
        else:
            act_sig_axes[model_sig].set_xticklabels(x_tick_labels)

    # restrict the area of the figures to leave TOP_MARGIN pixel free for title
    top_percentage = TOP_MARIGIN_PX / RES_HEIGHT
    # rect(left, bottom, right, top)
    plt.tight_layout(rect=(0, 0, 1, 1 - top_percentage))
    plt.subplots_adjust(hspace=0.4)

    global FRAMES
    # FRAMES = len(target_preds[0]) * ARTIFICIAL_FRAME_MULTIPLIER
    FRAMES = 20

    # ani = animation.FuncAnimation(fig, animate, frames=50,  # debug
    ani = animation.FuncAnimation(
        fig, animate, frames=FRAMES,
        fargs=(pred_artists, pred_data, pred_axes,
               act_data, tau_data, act_tau_artists, act_tau_axes,
               act_sig_artists, sig_data, act_sig_axes)
    )

    ani.save(filename='animated_freqpreds_3x4.mp4', dpi=DPI, fps=FPS)


# this function handles the update of every plot.
def animate(i,
            pred_artists, pred_data, pred_axes,
            act_data, tau_data, act_tau_artists, act_tau_axes,
            act_sig_artists, sig_data, act_sig_axes):
    global LATEST_DP

    # print progress
    print_progress(i, FRAMES, prefix="Progress",
                   suffix="Complete", bar_length=40)

    #
    # === UPDATE PREDICTION PLOTS ===
    #
    freqpred_x_inds = np.arange(0, 399)
    flat_artists = []
    if i % ARTIFICIAL_FRAME_MULTIPLIER == 0 and i > 0:  # here we only add a new datapoint every x frames
        LATEST_DP += 1
        x = freqpred_x_inds[0:LATEST_DP]

        for artists, pred_model in zip(pred_artists, pred_data):
            data = pred_data[pred_model]
            for neuron in range(len(artists)):
                # append new y
                activations = data[neuron][0:LATEST_DP]

                # set new data on the artist
                # not sure why there sometimes is a list of the artists and sometimes directly the artists...
                try:
                    artists[neuron].set_data(x, activations)
                    flat_artists.append(artists[neuron])
                except AttributeError:
                    artists[neuron][0].set_data(x, activations)
                    flat_artists.append(artists[neuron][0])

            # iterate over all artists on the axis and remove the previous timestep indicator vline
            for i in range(len(pred_axes[pred_model].get_lines())):
                if pred_axes[pred_model].lines[i].get_label() == "timestep_indicator":
                    pred_axes[pred_model].lines[i].remove()

            # draw a vertical line on the current time step
            pred_axes[pred_model].axvline(LATEST_DP, label="timestep_indicator", c="k", ls=":", zorder=-1)

    # update the x-lim on every frame so that we "slide" across the graph
    update_viewport(mode="art_frames", frame=i, viewport_width=VIEWPORT_WIDTH, x_ticks=freqpred_x_inds,
                    axes=pred_axes.values(), print=False, latest_dp=LATEST_DP,
                    art_frame_mult=ARTIFICIAL_FRAME_MULTIPLIER)

    #
    # === UPDATE ACTIVATION OVER TAU PLOTS ===
    #
    if i % ARTIFICIAL_FRAME_MULTIPLIER == 0 and i > 0:
        for model_tau in tau_data:

            # we only want to plot lstm and srn prediction plot, no activation over tau
            if model_tau == "mlstm" or model_tau == "msrn":
                continue

            artists = act_tau_artists[model_tau]
            model_xy = []

            # find unique tau vals
            distinct_tau = []
            for tau in tau_data[model_tau]:
                if tau not in distinct_tau:
                    distinct_tau.append(tau)

            distinct_tau.sort()

            # create dict mapping from tau vals to tau x indices
            tau_x_vals = {}
            tau_x_ind = 0
            for dist_tau in distinct_tau:
                tau_x_vals[dist_tau] = tau_x_ind
                tau_x_ind += 1

            for tau_ind, tau in enumerate(tau_data[model_tau]):
                # find tau value on axis
                axis_tau = tau_x_vals[tau]
                model_xy.append([axis_tau, act_data[model_tau][int(tau_ind), LATEST_DP]])

                # set new data on the artist
                # not sure why there sometimes is a list of the artists and sometimes directly the artists...
                try:
                    artists.set_offsets(model_xy)
                    flat_artists.append(artists)
                except AttributeError:
                    artists[0].set_offsets(model_xy)
                    flat_artists.append(artists[0])

    #
    # === UPDATE ACTIVATION OVER SIGMA PLOTS ===
    #
    if i % ARTIFICIAL_FRAME_MULTIPLIER == 0 and i > 0:
        for model_sig in sig_data:

            # we only want to plot lstm and srn prediction plot, no activation over sigma
            if model_sig == "mlstm" or model_sig == "msrn":
                continue

            #  we no longer want to plot this, instead we use the space to plot the lstm and srn activations
            if model_sig == "mctrnn" or model_sig == "mactrnn":
                continue

            artists = act_sig_artists[model_sig]
            model_xy = []
            # find unique sigma vals
            distinct_sig = []
            for sig in sig_data[model_sig]:
                if sig not in distinct_sig:
                    distinct_sig.append(sig)

            distinct_sig.sort()

            # create dict mapping from sigma vals to sigma x indices
            sig_x_vals = {}
            sig_x_ind = 0
            for dist_tau in distinct_sig:
                sig_x_vals[dist_tau] = sig_x_ind
                sig_x_ind += 1

            for sig_ind, sig in enumerate(sig_data[model_sig]):
                # find sig value on axis
                axis_sig = sig_x_vals[sig]
                model_xy.append([axis_sig, act_data[model_sig][int(axis_sig), LATEST_DP]])

                # set new data on the artist
                # not sure why there sometimes is a list of the artists and sometimes directly the artists...
                try:
                    artists.set_offsets(model_xy)
                    flat_artists.append(artists)
                except AttributeError as e:
                    print(e)
                    artists[0].set_offsets(model_xy)
                    flat_artists.append(artists[0])

    return flat_artists


if __name__ == "__main__":
    main()
