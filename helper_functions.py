import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# interpolates the original data, this makes the transition over the graph smoother
def interpolate_data(num_interpolations, original_data):
    extended_data = [[], []]
    for neuron in range(np.shape(original_data)[0]):
        tuple_ind = 0
        # minus 2 because we add one and np.shape() (399, x ) == (0 - 398, x)
        while tuple_ind <= np.shape(original_data)[1] - 2:
            # get current and current +1 datapoints for interpolation
            current, next = original_data[neuron][tuple_ind], original_data[neuron][tuple_ind + 1]
            additional_activations = np.linspace(current, next,
                                                 num_interpolations)  # this includes the original two data points
            for art_point in additional_activations:
                extended_data[neuron].append(art_point)
            tuple_ind += 2

        if np.shape(original_data)[1] % 2 != 0:
            # odd number of activations, do last step manually
            additional_activations = np.linspace(original_data[neuron][-2], original_data[neuron][-1],
                                                 num_interpolations)
            # remove the first activation value, that is already in the list from the last case above
            additional_activations = additional_activations[1:]
            for art_point in additional_activations:
                extended_data[neuron].append(art_point)

    return extended_data


# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    print('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        print()


# sets the xlim on a given axis, this is used to slide across the graph
def update_viewport(
        mode, frame, viewport_width, axes, x_ticks=None, latest_dp=None, art_frame_mult=None, print_lims=False, loc=None):
    for axis in axes:
        if mode == "art_frames":
            frame_step = 1 / art_frame_mult  # step we want to move per frame
            current_artificial_step = frame % art_frame_mult

            if loc == "right":
                x_min = latest_dp - viewport_width + (frame_step * current_artificial_step)
                x_max = latest_dp + viewport_width + (frame_step * current_artificial_step)

            elif loc is None:
                x_min = latest_dp - (viewport_width / 2) + (frame_step * current_artificial_step)
                x_max = latest_dp + (viewport_width / 2) + (frame_step * current_artificial_step)

        elif mode == "interpolation":
            x_min = x_ticks[frame] - (viewport_width / 2)
            x_max = x_ticks[frame] + (viewport_width / 2)

            if x_min < 0:
                x_min = 0

        if x_max < viewport_width:
            x_max = viewport_width

        # update the viewport
        if print_lims:
            print("Frame: {}, LATEST_DP: {}, xmin: {}, xmax: {}".format(frame, latest_dp, x_min, x_max))
        axis.set_xlim(x_min, x_max)
        axis.figure.canvas.draw()


# return the colors that 3b1b uses in his videos for the an dark theme
def get_tbob_colors():
    # taken from https://github.com/3b1b/manim/blob/master/manimlib/constants.py
    colors = {
        "DARK_BLUE": "#236B8E",
        "DARK_BROWN": "#8B4513",
        "LIGHT_BROWN": "#CD853F",
        "BLUE_E": "#1C758A",
        "BLUE_D": "#29ABCA",
        "BLUE_C": "#58C4DD",
        "BLUE_B": "#9CDCEB",
        "BLUE_A": "#C7E9F1",
        "TEAL_E": "#49A88F",
        "TEAL_D": "#55C1A7",
        "TEAL_C": "#5CD0B3",
        "TEAL_B": "#76DDC0",
        "TEAL_A": "#ACEAD7",
        "GREEN_E": "#699C52",
        "GREEN_D": "#77B05D",
        "GREEN_C": "#83C167",
        "GREEN_B": "#A6CF8C",
        "GREEN_A": "#C9E2AE",
        "YELLOW_E": "#E8C11C",
        "YELLOW_D": "#F4D345",
        "YELLOW_C": "#FFFF00",
        "YELLOW_B": "#FFEA94",
        "YELLOW_A": "#FFF1B6",
        "GOLD_E": "#C78D46",
        "GOLD_D": "#E1A158",
        "GOLD_C": "#F0AC5F",
        "GOLD_B": "#F9B775",
        "GOLD_A": "#F7C797",
        "RED_E": "#CF5044",
        "RED_D": "#E65A4C",
        "RED_C": "#FC6255",
        "RED_B": "#FF8080",
        "RED_A": "#F7A1A3",
        "MAROON_E": "#94424F",
        "MAROON_D": "#A24D61",
        "MAROON_C": "#C55F73",
        "MAROON_B": "#EC92AB",
        "MAROON_A": "#ECABC1",
        "PURPLE_E": "#644172",
        "PURPLE_D": "#715582",
        "PURPLE_C": "#9A72AC",
        "PURPLE_B": "#B189C6",
        "PURPLE_A": "#CAA3E8",
        "WHITE": "#FFFFFF",
        "BLACK": "#000000",
        "LIGHT_GRAY": "#BBBBBB",
        "LIGHT_GREY": "#BBBBBB",
        "GRAY": "#888888",
        "GREY": "#888888",
        "DARK_GREY": "#444444",
        "DARK_GRAY": "#444444",
        "DARKER_GREY": "#222222",
        "DARKER_GRAY": "#222222",
        "GREY_BROWN": "#736357",
        "PINK": "#D147BD",
        "GREEN_SCREEN": "#00FF00",
        "ORANGE": "#FF862F",
    }

    return colors


# return a dictionary of color configs, depending on the given color theme
def get_color_dict(theme):
    if theme == "sns":
        colors = sns.color_palette("muted")
        color_dict = {
            # general
            "fig_title": "k",
            "subplot_title": "k",
            # freqpred
            "neuron_0": colors[0],
            "neuron_1": colors[1],
            "freqpred_xlabel": "k",
            "freqpred_ylabel": "k",
            "freqpred_bg_0": colors[7],
            "freqpred_bg_1": colors[8],
            "timestep_indicator_0": "k",
            "timestep_indicator_1": "k",
            # tauplot
            "tauplot_xlabel": "k",
            "tauplot_ylabel": "k",
            "tauplot_scatter": colors[4],
            "tauplot_xtick_labels": "k",
            "tauplot_hbar": "k",
            # sigma plot
            "sigplot_xlabel": "k",
            "sigplot_ylabel": "k",
            "sigplot_scatter": colors[3],
            "sigplot_hbar": "k",
            "sigplot_xtick_labels": "k"
        }

    elif theme == "STH":
        colors = [(0.0, 0.0, 0.7), (1.0, 0.3, 0.3), (1.0, 0.0, 1.0), (0.0, 0.7, 0.7), (0.7, 0.7, 0.7), (0.5, 0.5, 0.7),
                  (0.7, 0.0, 0.0)]
        color_dict = {
            # general
            "fig_title": "k",
            "subplot_title": "k",
            # freqpred
            "neuron_0": colors[0],
            "neuron_1": colors[1],
            "freqpred_xlabel": "k",
            "freqpred_ylabel": "k",
            "freqpred_bg_0": "#595859",  # https://www.ibm.com/design/v1/language/resources/color-library/
            "freqpred_bg_1": "#fed500",
            "timestep_indicator_0": "k",
            "timestep_indicator_1": "k",
            # tauplot
            "tauplot_xlabel": "k",
            "tauplot_ylabel": "k",
            "tauplot_scatter": (8 / 255, 67 / 255, 132 / 255),  # color picked from pdf( darkest)
            "tauplot_xtick_labels": "k",
            "tauplot_hbar": "k",
            # sigma plot
            "sigplot_xlabel": "k",
            "sigplot_ylabel": "k",
            "sigplot_scatter": (80 / 255, 0, 107 / 255),  # color picked from pdf( darkest)
            "sigplot_hbar": "k",
            "sigplot_xtick_labels": "k",
            "highglight_red": colors[6]
        }

    elif theme == "STH_dark":
        # to make a color value brighter, this gives us the new values:
        # https://graphicdesign.stackexchange.com/questions/75417/how-to-make-a-given-color-a-bit-darker-or-lighter
        brighter_percentage = 0.5
        colors = [make_color_brighter((0.0, 0.0, 0.7), brighter_percentage),
                  make_color_brighter((1.0, 0.3, 0.3), brighter_percentage), (1.0, 0.0, 1.0), (0.0, 0.7, 0.7),
                  (0.7, 0.7, 0.7), (0.5, 0.5, 0.7), (0.7, 0.0, 0.0)]
        color_dict = {
            # general
            "fig_title": "w",
            "subplot_title": "w",
            # freqpred
            "neuron_0": colors[0],
            "neuron_1": colors[1],
            "freqpred_xlabel": "w",
            "freqpred_ylabel": "w",
            "freqpred_bg_0": "#595859",  # https://www.ibm.com/design/v1/language/resources/color-library/
            "freqpred_bg_1": colors[4],
            "timestep_indicator_0": "w",
            "timestep_indicator_1": "w",
            # tauplot
            "tauplot_xlabel": "w",
            "tauplot_ylabel": "w",
            # color picked from pdf( darkest)
            "tauplot_scatter": make_color_brighter((8, 67, 132), brighter_percentage),
            "tauplot_xtick_labels": "w",
            "tauplot_hbar": "w",
            # sigma plot
            "sigplot_xlabel": "w",
            "sigplot_ylabel": "w",
            # color picked from pdf( darkest)
            "sigplot_scatter": make_color_brighter((79, 0, 107), brighter_percentage),
            "sigplot_hbar": "w",
            "sigplot_xtick_labels": "w",
            "highglight_red": make_color_brighter((0.7, 0.0, 0.0), brighter_percentage)
        }

    elif theme == "3b1b":
        colors = get_tbob_colors()
        color_dict = {
            # general
            "fig_title": "w",
            "subplot_title": "w",
            # freqpred
            "neuron_0": colors["BLUE_C"],
            "neuron_1": colors["PURPLE_C"],
            "freqpred_xlabel": "w",
            "freqpred_ylabel": "w",
            "freqpred_bg_0": "k",
            "freqpred_bg_1": colors["GREY"],
            "timestep_indicator_0": "w",
            "timestep_indicator_1": "w",
            # tauplot
            "tauplot_xlabel": "w",
            "tauplot_ylabel": "w",
            "tauplot_scatter": colors["GREEN_C"],
            "tauplot_xtick_labels": "w",
            "tauplot_hbar": "w",
            # sigma plot
            "sigplot_xlabel": "w",
            "sigplot_ylabel": "w",
            "sigplot_scatter": colors["RED_C"],
            "sigplot_hbar": "w",
            "sigplot_xtick_labels": "w"
        }

    else:
        raise LookupError("No color match color configuration for theme '" + theme + "'")

    return color_dict


# shows a plot with all the 3b1b colors
def show_tbob_color_plot():
    # get 3b1b colors
    color_dict = get_tbob_colors()

    # use dark style
    plt.style.use("dark_background")

    # plot the colors as horizontal bars
    fig, ax = plt.subplots()

    color_names = list(color_dict.keys())
    color_names.sort()
    color_vals = [color_dict[color] for color in color_names]
    y_inds = np.arange(len(color_names))
    x_length = 10

    ax.barh(y_inds, x_length, align="center", color=color_vals)
    ax.set_yticks(y_inds)
    ax.set_yticklabels(color_names, fontsize="5")
    ax.set_title("3Blue1Brown color dict with MPL dark style")

    plt.savefig("3B1B_colors.png", dpi=400)
    plt.show()


# shows a plots with all the colors from STH's palette (code/tools/utils_plot.py line 25)
def show_sth_color_plot():
    colors = [(0.0, 0.0, 0.7), (1.0, 0.3, 0.3), (1.0, 0.0, 1.0), (0.0, 0.7, 0.7), (0.7, 0.7, 0.7), (0.5, 0.5, 0.7)]

    # plot the colors as horizontal bars
    fig, ax = plt.subplots()
    color_names = np.arange(len(colors))
    y_inds = np.arange(len(colors))
    x_length = 10

    ax.barh(y_inds, x_length, align="center", color=colors)
    ax.set_yticks(y_inds)
    ax.set_yticklabels(color_names, fontsize="5")
    ax.set_title("Stefan Heinrich's color palette")

    plt.savefig("STH_colors.png", dpi=400)
    plt.show()


# helper to calculate brighter versions of a color
def make_color_brighter(color, percentage):
    # https://graphicdesign.stackexchange.com/questions/75417/how-to-make-a-given-color-a-bit-darker-or-lighter
    brighter_color = []
    if max(color) <= 1:
        # convert to 0 to 255 range
        color = [component * 255 for component in color]
    for index, component in enumerate(color):
        diff = abs(255 - component)
        add_value = diff * percentage
        brighter_component = component + add_value
        brighter_color.append(brighter_component)

    # convert back to 0-1 range
    brighter_color = [component / 255 for component in brighter_color]

    return tuple(brighter_color)


if __name__ == "__main__":
    show_tbob_color_plot()
    show_sth_color_plot()
    # make_color_brighter((1.0, 0.3, 0.3), 0.5)


