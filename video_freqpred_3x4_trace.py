import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helper_functions import update_viewport
from helper_functions import print_progress
from helper_functions import get_color_dict
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
TRACE_DECAY_GAMMA = 0.5
C_THEME = "STH_dark"
CPAL = get_color_dict(C_THEME)


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
        with np.load("eval/freqpred/vctrnn_freqpred_" + name_id[0] + name_id[1] + "_tausigmavals.npz") as f:
            tau_sigma = f['arr_0']

            # srn and lstm dont have tau and sigma data...
            if name_id[0] != 'lstm' and name_id[0] != 'srn':
                sigma = tau_sigma[-1, 1, :]
                # if name_id[0] == "mvctrnn":
                #     print(sigma[-1])
                tau = tau_sigma[-1, 0, :]

                tau_data[name_id[0]] = tau
                sig_data[name_id[0]] = sigma

        # here we add an m to the name, so that we can later remove the trailing "m" for EVERY model name
        if name_id[0] == "lstm":
            name = "mlstm"
        elif name_id[0] == "srn":
            name = "msrn"
        else:
            name = name_id[0]

        # create a dict for every model
        load_model = {
            "name": name,
            "preds": np.transpose(
                np.load("eval/freqpred/vctrnn_freqpred_"
                        + name_id[0] + name_id[1]
                        + "_predvals.npz")['arr_0'][0][:][:]),
            "hid_acts": np.transpose(
                np.load("eval/freqpred/vctrnn_freqpred_"
                        + name_id[0] + name_id[1]
                        + "_hidavals.npz")['arr_0'][0][:][:]),
            "taus": tau,
            "sigmas": sigma
        }

        # save the complete dict
        model_data.append(load_model)
        act_data[name] = load_model["hid_acts"]
        pred_data[name] = load_model["preds"]

    return model_data, pred_data, act_data, tau_data, sig_data


# add every axes to the figure
def add_axes(fig):

    # create an axis for every subplot
    ax_ctrnn_acts_tau = fig.add_subplot(3, 4, 1)
    ax_actrnn_acts_tau = fig.add_subplot(3, 4, 2)
    ax_vctrnn_acts_tau = fig.add_subplot(3, 4, 4)
    ax_avctrnn_acts_tau = fig.add_subplot(3, 4, 3)

    ax_ctrnn_preds = fig.add_subplot(3, 4, 5)
    ax_actrnn_preds = fig.add_subplot(3, 4, 6)
    ax_vctrnn_preds = fig.add_subplot(3, 4, 8)
    ax_avctrnn_preds = fig.add_subplot(3, 4, 7)

    ax_lstm_preds = fig.add_subplot(3, 4, 9)
    ax_srn_preds = fig.add_subplot(3, 4, 10)
    ax_vctrnn_acts_sig = fig.add_subplot(3, 4, 12)
    ax_avctrnn_axts_sig = fig.add_subplot(3, 4, 11)

    # for later iteration
    axes = [ax_ctrnn_preds, ax_actrnn_preds, ax_vctrnn_preds, ax_avctrnn_preds, ax_ctrnn_acts_tau, ax_actrnn_acts_tau,
            ax_vctrnn_acts_tau, ax_avctrnn_acts_tau, ax_lstm_preds, ax_srn_preds, ax_vctrnn_acts_sig,
            ax_avctrnn_axts_sig]

    # create group of the relating axis, we will iterate over those groups to animate the specific plots
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

    act_sig_axes = {"mvctrnn": ax_vctrnn_acts_sig,
                    "mavctrnn": ax_avctrnn_axts_sig}

    return axes, pred_axes, act_tau_axes, act_sig_axes


# this methods takes care of setting appropriate ticks and axis limits for the activation over sigma and activation
# over tau plots. For the prediction plots, this was much less complex and doesn't require a dedicated method
def setup_ticks(data, axes, plot_type):
    # setup xmin xmax and ticks
    for model in data:

        # because we have more sigma data than we actually plot...
        if plot_type == "sigma":
            # we only want to plot lstm and srn prediction plot, no activation over sigma
            if model == "mlstm" or model == "msrn":
                continue

            #  we no longer want to plot this, instead we use the space to plot the lstm and srn activations
            if model == "mctrnn" or model == "mactrnn":
                continue

        # find unique tau vals
        distinct_vals = []
        for val in data[model]:
            if val not in distinct_vals:
                distinct_vals.append(val)

        distinct_vals.sort()

        # create dict mapping from tau vals to tau x indices
        tau_x_vals = {}
        tau_x_ind = 0
        for dist_tau in distinct_vals:
            tau_x_vals[dist_tau] = tau_x_ind
            tau_x_ind += 1

        # set x ticks
        x_ticks = list(tau_x_vals.values())
        x_ticks.sort()
        axes[model].set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
        axes[model].set_xticks(x_ticks)

        # set x ticks labels
        x_tick_labels = [str(np.around(tau, decimals=2)) for tau in distinct_vals]
        # if we have many x vals, only display four values: first, last and two in-between
        if len(x_tick_labels) > 5:
            for j in range(len(x_tick_labels)):
                # here, we only display four x_tick_lables, thus we set empty strings as every other tick label
                inds = [int(np.around(ind)) for ind in np.linspace(0, len(x_tick_labels) - 1, 4)]
                if j in inds:
                    pass
                else:
                    x_tick_labels[j] = ""

            axes[model].set_xticklabels(
                x_tick_labels,
                color=CPAL["tauplot_xtick_labels"],
                fontsize=TICKS_FS)

        else:
            axes[model].set_xticklabels(
                x_tick_labels,
                color=CPAL["tauplot_xtick_labels"],
                fontsize=TICKS_FS)

        # adjust y-ticks to only have one decimal, to decrease width with larget font size
        y_tick_labels = axes[model].get_yticks()
        y_tick_labels = [np.around(tick, decimals=1) for tick in y_tick_labels]
        axes[model].set_yticklabels(y_tick_labels, fontsize=TICKS_FS)


def main():
    # load data
    models, pred_data, act_data, tau_data, sig_data = load_data()

    if (C_THEME == "3b1b") or (C_THEME == "STH_dark"):
        plt.style.use("dark_background")

    # setup figure
    fig = plt.figure(figsize=(RES_WIDTH / DPI, RES_HEIGHT / DPI))
    fig.suptitle("Adaptive and Variational CTRNN's: Human Motion Patterns Prediction",
                 fontfamily="sans-serif",
                 fontweight="bold",
                 fontsize=FIG_FS,
                 color=CPAL["fig_title"])

    # add axes for each plot
    axes, pred_axes, act_tau_axes, act_sig_axes = add_axes(fig)

    # set axis titles
    for model in models:
        # set titles on predictions plots
        for key in pred_axes.keys():
            if model["name"] == key:
                model_name = model["name"][1:]  # remove the trailing "m"
                pred_axes[key].set_title(
                    model_name.upper() + " predictions", fontsize=SUBPLOT_TITLE_FS, color=CPAL["subplot_title"])
                pred_axes[key].set_xlabel("Timestep", fontsize=XLABEL_FS, color=CPAL["freqpred_xlabel"])
                pred_axes[key].set_ylabel("Hidden actiation", fontsize=YLABEL_FS, color=CPAL["freqpred_ylabel"])
                # set fontsize of ticks
                pred_axes[key].tick_params(labelsize=TICKS_FS)

        # set titles on activation over tau plots
        for key in act_tau_axes.keys():
            if model["name"] == key:
                model_name = model["name"][1:]  # remove the trailing "m"
                act_tau_axes[key].set_title(
                    model_name.upper() + " activation over timescale", fontsize=SUBPLOT_TITLE_FS,
                    color=CPAL["subplot_title"])
                act_tau_axes[key].set_xlabel(r"Timescale $\tau$", fontsize=XLABEL_FS, color=CPAL["tauplot_xlabel"])
                act_tau_axes[key].set_ylabel("Hidden activation", fontsize=YLABEL_FS, color=CPAL["tauplot_ylabel"])
                pred_axes[key].tick_params(labelsize=TICKS_FS)

        # set titles on activation over sigma plots
        for key in act_sig_axes.keys():
            if model["name"] == key:
                model_name = model["name"][1:]  # remove the trailing "m"
                act_sig_axes[key].set_title(
                    model_name.upper() + " activation over variance", fontsize=SUBPLOT_TITLE_FS,
                    color=CPAL["subplot_title"])
                act_sig_axes[key].set_xlabel(r"Variance $\sigma$", fontsize=XLABEL_FS, color=CPAL["sigplot_xlabel"])
                act_sig_axes[key].set_ylabel("Hidden activation", fontsize=YLABEL_FS, color=CPAL["sigplot_ylabel"])
                pred_axes[key].tick_params(labelsize=TICKS_FS)

    # set y-lim on every axis
    for axis in axes:
        axis.set_ylim(-1, 1)

    # init line artists
    # row 1: prediction plots
    ctrnn_pred_n0 = pred_axes["mctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_0"])
    ctrnn_pred_n1 = pred_axes["mctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_1"])

    actrnn_pred_n0 = pred_axes["mactrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_0"])
    actrnn_pred_n1 = pred_axes["mactrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_1"])

    vctrnn_pred_0 = pred_axes["mvctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_0"])
    vctrnn_pred_1 = pred_axes["mvctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_1"])

    avctrnn_pred_0 = pred_axes["mavctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_0"])
    avctrnn_pred_1 = pred_axes["mavctrnn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_1"])

    lstm_pred_0 = pred_axes["mlstm"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_0"])
    lstm_pred_1 = pred_axes["mlstm"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_1"])

    srn_pred_0 = pred_axes["msrn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_0_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_0"])
    srn_pred_1 = pred_axes["msrn"].plot(
        [], [], lw=NEURONS_LINE_WIDTH, label=NEURON_1_LABEL, ls=NEURONS_LINE_STYLE, c=CPAL["neuron_1"])

    # for later iteration
    #pred_artists = [
    #    [ctrnn_pred_n0, ctrnn_pred_n1], [actrnn_pred_n0, actrnn_pred_n1], [vctrnn_pred_0, vctrnn_pred_1],
    #    [avctrnn_pred_0, avctrnn_pred_1], [lstm_pred_0, lstm_pred_1], [srn_pred_0, srn_pred_1]]

    pred_artists = {
        "mctrnn": [ctrnn_pred_n0, ctrnn_pred_n1],
        "mactrnn": [actrnn_pred_n0, actrnn_pred_n1],
        "mvctrnn": [vctrnn_pred_0, vctrnn_pred_1],
        "mavctrnn": [avctrnn_pred_0, avctrnn_pred_1],
        "mlstm": [lstm_pred_0, lstm_pred_1],
        "msrn": [srn_pred_0, srn_pred_1]
    }

    # load target preds outside of loops
    target_preds = np.transpose(np.load('eval/freqpred/vctrnn_freqpred_test_targetvals.npz')['arr_0'][0, :, :])

    # draw gt prediction for pred plots
    gt_x = np.arange(0, np.shape(target_preds)[1])
    gt_y0 = target_preds[0]
    gt_y1 = target_preds[1]
    for ax_pred_plot in pred_axes.values():
        # if we use the dark color theme, we need to adjust the alpha of the first half of the ground truth,
        # because alpha < 1 in front of black makes it even harder to see...
        if C_THEME == "3b1b":
            # plot first half of gt with increased alpha
            ax_pred_plot.plot(gt_x[0:201], gt_y0[0:201], c=CPAL["neuron_0"], alpha=0.5, lw=GT_LINE_WIDTH,
                              label=GT_0_LABEL)
            ax_pred_plot.plot(gt_x[0:201], gt_y1[0:201], c=CPAL["neuron_1"], alpha=0.5, lw=GT_LINE_WIDTH,
                              label=GT_1_LABEL)

            # plot second half with normal alpha, because background is brigther here
            ax_pred_plot.plot(gt_x[200:], gt_y0[200:], c=CPAL["neuron_0"], alpha=0.3, lw=GT_LINE_WIDTH,
                              label=GT_0_LABEL)
            ax_pred_plot.plot(gt_x[200:], gt_y1[200:], c=CPAL["neuron_1"], alpha=0.3, lw=GT_LINE_WIDTH,
                              label=GT_1_LABEL)

        else:
            ax_pred_plot.plot(gt_x, gt_y0, c=CPAL["neuron_0"], alpha=0.3, lw=GT_LINE_WIDTH, label=GT_0_LABEL)
            ax_pred_plot.plot(gt_x, gt_y1, c=CPAL["neuron_1"], alpha=0.3, lw=GT_LINE_WIDTH, label=GT_1_LABEL)

        ax_pred_plot.axvspan(0, 200, facecolor=CPAL["freqpred_bg_0"], alpha=0.3)
        # ax_pred_plot.axvspan(0, 200, facecolor="w", alpha=0.8, zorder=-1)
        ax_pred_plot.axvspan(200, 450, facecolor=CPAL["freqpred_bg_1"], alpha=0.3)
        # this makes sure that this we have a white background when we use the dark style
        # ax_pred_plot.axvspan(200, 450, facecolor="w", alpha=1, zorder=-1)

    # row 2: hidden activations over tau
    ctrnn_act_tau = act_tau_axes["mctrnn"].scatter([], [], c=CPAL["tauplot_scatter"])

    actrnn_act_tau = act_tau_axes["mactrnn"].scatter([], [], c=CPAL["tauplot_scatter"])
    act_tau_axes["mactrnn"].set_xticklabels([123.45])

    vctrnn_act_tau = act_tau_axes["mvctrnn"].scatter([], [], c=CPAL["tauplot_scatter"])

    avctrnn_act_tau = act_tau_axes["mavctrnn"].scatter([], [], c=CPAL["tauplot_scatter"])
    act_tau_axes["mavctrnn"].set_xticklabels([123.45])

    act_tau_artists = {
        "mctrnn": ctrnn_act_tau,
        "mactrnn": actrnn_act_tau,
        "mvctrnn": vctrnn_act_tau,
        "mavctrnn": avctrnn_act_tau
    }

    # add horizontal bar on 0 to all activation plots
    for axis in act_tau_axes.values():
        axis.axhline(c=CPAL["tauplot_hbar"], zorder=-1, lw=1)

    # set the ticks and lims on the activation over tau plots
    setup_ticks(tau_data, act_tau_axes, plot_type="tau")

    # row 3: hidden activations over sigma for vctrnn and avctrnn (lstm and srn preds have been added earlier)
    vctrnn_act_sig = act_sig_axes["mvctrnn"].scatter([], [], c=CPAL["sigplot_scatter"])
    avctrnn_act_sig = act_sig_axes["mavctrnn"].scatter([], [], c=CPAL["sigplot_scatter"])

    act_sig_artists = {
        "mvctrnn": vctrnn_act_sig,
        "mavctrnn": avctrnn_act_sig
    }

    # add horizontal bar on 0 to all activation plots
    for axis in act_sig_axes.values():
        axis.axhline(c=CPAL["sigplot_hbar"], zorder=-1, lw=1)

    # set the ticks and lims on the activation over tau plots
    setup_ticks(sig_data, act_sig_axes, plot_type="sigma")

    # restrict the area of the figures to leave TOP_MARGIN pixel free for title
    top_percentage = TOP_MARIGIN_PX / RES_HEIGHT
    # rect(left, bottom, right, top)
    plt.tight_layout(rect=(0, 0, 1, 1 - top_percentage))
    plt.subplots_adjust(hspace=0.4)

    global FRAMES
    FRAMES = len(target_preds[0]) * ARTIFICIAL_FRAME_MULTIPLIER
    # FRAMES = 20

    # ani = animation.FuncAnimation(fig, animate, frames=50,  # debug
    ani = animation.FuncAnimation(
        fig, animate, frames=FRAMES,
        fargs=(pred_artists, pred_data, pred_axes,
               act_data, tau_data, act_tau_artists, act_tau_axes,
               act_sig_artists, sig_data, act_sig_axes)
    )

    ani.save(filename='animated_freqpreds_3x4_trace.mp4', dpi=DPI, fps=FPS)


# this function handles the update of the plots on every frame
def animate(i,
            pred_artists, pred_data, pred_axes,
            act_data, tau_data, act_tau_artists, act_tau_axes,
            act_sig_artists, sig_data, act_sig_axes):
    global LATEST_DP

    # prints progressbar to cli
    print_progress(i, FRAMES, prefix="Progress", suffix="Complete", bar_length=40)

    #
    # === UPDATE PREDICTION PLOTS ===
    #
    freqpred_x_inds = np.arange(0, 399)
    flat_artists = []
    if i % ARTIFICIAL_FRAME_MULTIPLIER == 0 and i > 0:  # here we only add a new datapoint every x frames
        LATEST_DP += 1
        x = freqpred_x_inds[0:LATEST_DP]

        # for artists, pred_model in zip(pred_artists, pred_data):
        for pred_model in pred_data:
            artists = pred_artists[pred_model]

            data = pred_data[pred_model]
            for neuron in range(len(artists)):
                # append new y
                activations = data[neuron][0:LATEST_DP]

                # set new data on the artist
                # not sure why there sometimes is a list of the artists and sometimes directly the artists...
                try:
                    artists[neuron].set_data(x, activations)
                    flat_artists.append(artists[neuron])

                    # sanity check: model name should be in title of subplot
                    subplot_title = artists[neuron].axes.title._text.lower()
                    assert_msg = "Model name '{}' does not appear to occur in subplot title '{}'".format(
                        pred_model, subplot_title)
                    assert pred_model[1:] in subplot_title, assert_msg  # cut preceding "m" from model name ...

                except AttributeError:
                    artists[neuron][0].set_data(x, activations)
                    flat_artists.append(artists[neuron][0])

                    # sanity check: model name should be in title of subplot
                    subplot_title = artists[neuron][0].axes.title._text.lower()
                    assert_msg = "Model name '{}' does not appear to occur in subplot title '{}'".format(
                                pred_model, subplot_title)
                    assert pred_model[1:] in subplot_title, assert_msg  # cut preceding "m" from model name ...

    # update the time indicator on every frame, not just when we update the plot data
    for pred_model in pred_data:
        # iterate over all artists on the axis and remove the previous timestep indicator vline
        for j in range(len(pred_axes[pred_model].get_lines())):
            if pred_axes[pred_model].lines[j].get_label() == "timestep_indicator":
                pred_axes[pred_model].lines[j].remove()

        # draw a vertical line on the current time step
        if LATEST_DP >= 199:
            vline_col = CPAL["timestep_indicator_1"]
        else:
            vline_col = CPAL["timestep_indicator_0"]

        frame_step = 1 / ARTIFICIAL_FRAME_MULTIPLIER  # step we want to move per frame
        current_artificial_step = i % ARTIFICIAL_FRAME_MULTIPLIER  # current step between two natural frames
        # the time location taking the artificial frames into account
        artificial_time_loc = LATEST_DP + (frame_step * current_artificial_step)

        pred_axes[pred_model].axvline(
            artificial_time_loc, label="timestep_indicator", c=vline_col, ls=":")

    # update the x-lim on every frame so that we "slide" across the graph
    update_viewport(mode="art_frames", frame=i, viewport_width=VIEWPORT_WIDTH, x_ticks=freqpred_x_inds,
                    axes=pred_axes.values(), latest_dp=LATEST_DP,
                    art_frame_mult=ARTIFICIAL_FRAME_MULTIPLIER, print_lims=False)

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
            all_xy = []
            all_colors = []

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

            # create color and points for current timestep
            for tau_ind, tau in enumerate(tau_data[model_tau]):
                # find sig value on axis
                axis_tau = tau_x_vals[tau]

                all_xy.append([axis_tau, act_data[model_tau][int(axis_tau), LATEST_DP]])
                all_colors.append(CPAL["tauplot_scatter"])

            # create color and points for the trace
            for tau_ind, tau in enumerate(tau_data[model_tau]):
                for trace_step in range(1, LATEST_DP):
                    # this should be enough for everything ti fully decay
                    if trace_step >= 10:
                        break
                    axis_tau = tau_x_vals[tau]

                    new_alpha = TRACE_DECAY_GAMMA ** trace_step

                    new_rgba = list(mpl.colors.to_rgb(CPAL["tauplot_scatter"]))
                    new_rgba.append(new_alpha)

                    all_xy.append([axis_tau, act_data[model_tau][int(axis_tau), LATEST_DP - trace_step]])
                    all_colors.append(new_rgba)

            # now that we have current and trace data, we update the artists
            assert len(all_colors) == len(all_xy), "these must have the same length"
            try:
                artists.set_offsets(all_xy)
                artists.set_color(all_colors)
                flat_artists.append(artists)
            except AttributeError as e:
                print(e)
                artists[0].set_offsets(all_xy)
                artists[0].set_color(all_colors)
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
            all_xy = []
            all_colors = []

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

            # create color and points for current timestep
            for sig_ind, sig in enumerate(sig_data[model_sig]):
                # find sig value on axis
                axis_sig = sig_x_vals[sig]

                all_xy.append([axis_sig, act_data[model_sig][int(axis_sig), LATEST_DP]])
                all_colors.append(CPAL["sigplot_scatter"])

            # create color and points for the trace
            for sig_ind, sig in enumerate(sig_data[model_sig]):
                for trace_step in range(1, LATEST_DP):
                    # this should be enough for everything ti fully decay
                    if trace_step >= 10:
                        break
                    axis_sig = sig_x_vals[sig]

                    new_alpha = TRACE_DECAY_GAMMA ** trace_step

                    new_rgba = list(mpl.colors.to_rgb(CPAL["sigplot_scatter"]))
                    new_rgba.append(new_alpha)

                    all_xy.append([axis_sig, act_data[model_sig][int(axis_sig), LATEST_DP - trace_step]])
                    all_colors.append(new_rgba)

            # now that we have current and trace data, we update the artists
            assert len(all_colors) == len(all_xy), "these must have the same length"
            try:
                artists.set_offsets(all_xy)
                artists.set_color(all_colors)
                flat_artists.append(artists)
            except AttributeError as e:
                print(e)
                artists[0].set_offsets(all_xy)
                artists[0].set_color(all_colors)
                flat_artists.append(artists[0])

    return flat_artists


if __name__ == "__main__":
    main()
