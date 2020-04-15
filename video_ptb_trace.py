import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import os
import sys
import seaborn as sns
from matplotlib.colors import ListedColormap
from helper_functions import print_progress
from helper_functions import get_color_dict

# from matplotlib import rc
# rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{xcolor}")

# some globals
MAIN_FOLDER = "./ptb"
BATCHID = 19
TOP_MARIGIN = 60
# BOT_MARGIN = 40
RES_WIDTH = 1920
RES_HEIGHT = 1080
DPI = 100
ART_TEXT_MARGIN = 0.0035  # controls how much spacing is between the the current word and the surrounding text artists
FPS = 1
NUM_PREV_WORDS = 3
NUM_FOLLOWING_WORDS = 3
pal = sns.cubehelix_palette(100)
ANNOTATION = (r"consumers may want to move their telephones a little closer to the tv set <unk> <unk> watching abc 's "
              r"monday night football can now vote during <unk> for the greatest play in N years from among four or "
              r"five <unk> <unk>")

# removed the <unk>'s and made abc 's to abc's, so that num frames eq num words
ANNOTATION_ADJUSTED = (r"consumers may want to move their telephones a little closer to the tv set watching abc's "
                       r"monday night football can now vote during for the greatest play in N years from among "
                       r"four or five")
XLIM = 12.8
# make own cmap from seaborn palette
SNS_COLORS = ListedColormap(sns.color_palette(sns.cubehelix_palette(100, reverse=True)).as_hex())
TRACE_ACTIVATIONS = []
TRACE_COLORS = []
TRACE_DECAY_GAMMA = 0.25
C_THEME = "STH"
C_PAL = get_color_dict(C_THEME)

# normal fonts:
# ANNOTATION_FS = 15
# XLABEL_FS = 10
# YLABEL_FS = 10
# SUBPLOT_TITLE_FS = 13
# FIG_FS = 18
# BOT_MARGIN = 40
# TICKS_FS = 10

# big fonts:
ANNOTATION_FS = 15
XLABEL_FS = 13
YLABEL_FS = 13
SUBPLOT_TITLE_FS = 15
FIG_FS = 18
TICKS_FS = 13
BOT_MARGIN = 60

# get all models and load data
models = {}
timesteps = 0
for folder in os.listdir(MAIN_FOLDER):  # for each model...
    params = folder.split('_')
    model_dict = {}
    model_path = os.path.join(MAIN_FOLDER, folder)

    if os.path.isdir(model_path):
        for file in os.listdir(model_path):  # for each file in model folder
            if "hidavals.npz" in file:
                hidden_activations = np.load(os.path.join(model_path, file))['arr_0']
                act = np.asarray(hidden_activations)[0, :, BATCHID, :]
                model_dict["hidac_vals"] = act
                timesteps = model_dict["hidac_vals"].shape[0]
            elif "tausigmavals.npz" in file:
                model_dict["tau_vals"] = np.load(os.path.join(model_path, file))['arr_0'][-1, 0, :]
                model_dict["sigma_vals"] = np.load(os.path.join(model_path, file))['arr_0'][-1, 1, :]

    models[params[0]] = model_dict  # save in dict

if (C_THEME == "3b1b") or (C_THEME == "STH_dark"):
    plt.style.use("dark_background")

# setup plot grid (2x2 tiles: MCTRNN, MACTRNN / MAVTRNN, MVTRNN)
fig = plt.figure(figsize=(RES_WIDTH / DPI, RES_HEIGHT / DPI))  # 960 x 500 px per scatter
ax_mctrnn = fig.add_subplot(221)
ax_mactrnn = fig.add_subplot(222)
ax_mvctrnn = fig.add_subplot(223)
ax_mavctrnn = fig.add_subplot(224)

# for later iteration
axes = [ax_mctrnn, ax_mactrnn, ax_mvctrnn, ax_mavctrnn]

# set titles and prepare animated text
fig.suptitle("Adaptive and Variational CTRNN's: Penn TreeBank Language Modelling",
             fontfamily="sans-serif", fontweight="bold", fontsize=FIG_FS)
ax_mctrnn.set_title("CTRNN", fontsize=SUBPLOT_TITLE_FS)
ax_mactrnn.set_title("ACTRNN", fontsize=SUBPLOT_TITLE_FS)
ax_mavctrnn.set_title("AVCTRNN", fontsize=SUBPLOT_TITLE_FS)
ax_mvctrnn.set_title("VCTRNN", fontsize=SUBPLOT_TITLE_FS)

# all the stuff that is the same for all subplots
for axis in axes:
    # axis.set_xlim(0, max(models["mctrnn"]["tau_vals"]))
    axis.set_xlim(0, XLIM)
    axis.set_ylim(0, 1)
    axis.set_xlabel(r"Hidden unit's timescale $\tau$", fontsize=XLABEL_FS)
    axis.set_ylabel("Hidden activation", fontsize=YLABEL_FS)
    axis.tick_params(labelsize=TICKS_FS)

# global init of the artists
mctrnn_scatter = ax_mctrnn.scatter([], [])
mactrnn_scatter = ax_mactrnn.scatter([], [])
mavctrnn_scatter = ax_mavctrnn.scatter([], [])
mvctrnn_scatter = ax_mvctrnn.scatter([], [])

# rect "squeezes" all the subplots into the figure area specified by the rect
top_percentage = TOP_MARIGIN / RES_HEIGHT
bot_percentage = BOT_MARGIN / RES_HEIGHT
# rect(left, bottom, right, top)
plt.tight_layout(rect=(0, bot_percentage, 1, 1 - top_percentage))
plt.subplots_adjust(hspace=0.3, wspace=0.13)

#
# === BOTTOM ANNOTATION TEXT === #
#
# color the current word, we need to handle each word individually, because of the coloring
# words = ANNOTATION_ADJUSTED.split(" ")
words = ANNOTATION.split(" ")
row_1_words = words[0:16]
row_2_words = words[16:]

# find the starting location for the first and second row, by getting the width of the two lines
row_1_text = plt.figtext(
    0.5, 0.01, fontsize=ANNOTATION_FS, ha="center", s=" ".join(row_1_words), color=(0, 0, 0, 0))
row_2_text = plt.figtext(
    0.5, 0.005, fontsize=ANNOTATION_FS, ha="center", s=" ".join(row_2_words), color=(0, 0, 0, 0))

# get data in display resolution
row_1_text.draw(fig.canvas.get_renderer())
row_1_bb = row_1_text.get_window_extent()

row_2_text.draw(fig.canvas.get_renderer())
row_2_bb = row_2_text.get_window_extent()

# convert to figure resolution
row_1_start_x = row_1_bb.xmin / RES_WIDTH
row_2_start_x = row_2_bb.xmin / RES_WIDTH

# set the color of the current word
# we use the same color dict as in 3x4, I just used color fields that works in dark and bright...
color_list = [mpl.colors.to_rgb(C_PAL["fig_title"]) for word in words]

# because we use the bounding boxes of the individual words, its easiest to directly add a space to each word
# instead of adding synthetic margin between the bounding boxes of two words
row_1_words = [word + " " for word in row_1_words]
row_2_words = [word + " " for word in row_2_words]

# we will iterate over those and change the color of the arists
word_arists = []

for j in range(len(row_1_words)):
    if j == 0:
        text = plt.figtext(row_1_start_x, 0.03, row_1_words[j], color=color_list[j], fontsize=ANNOTATION_FS)
    else:
        text = plt.figtext(
            row_1_start_x, 0.03, row_1_words[j], color=color_list[j], transform=t, fontsize=ANNOTATION_FS)

    text.draw(fig.canvas.get_renderer())
    ex = text.get_window_extent()
    t = mpl.transforms.offset_copy(text._transform, x=ex.width, units="dots")
    word_arists.append(text)

for j in range(len(row_2_words)):
    if j == 0:
        text = plt.figtext(
            row_2_start_x, 0.01, row_2_words[j], color=color_list[len(row_1_words) + j], fontsize=ANNOTATION_FS)
    else:
        text = plt.figtext(
            row_2_start_x, 0.01, row_2_words[j], color=color_list[len(row_1_words) + j], transform=t,
            fontsize=ANNOTATION_FS)

    text.draw(fig.canvas.get_renderer())
    ex = text.get_window_extent()
    t = mpl.transforms.offset_copy(text._transform, x=ex.width, units="dots")
    word_arists.append(text)


# init function needed for blitting and passing references
def init():
    return mctrnn_scatter, mactrnn_scatter, mavctrnn_scatter, mvctrnn_scatter


# the update function
def update_data(i):
    global TRACE_COLORS
    global TRACE_ACTIVATIONS

    print_progress(i, timesteps, prefix="Progress", suffix="Complete", bar_length=40)

    mctrnn_xy = []
    mactrnn_xy = []
    mavctrnn_xy = []
    mvctrnn_xy = []

    mctrnn_colors = []
    mactrnn_colors = []
    mavctrnn_colors = []
    mvctrnn_colors = []

    trace_act_dict = {"mctrnn": [], "mactrnn": [], "mavctrnn": [], "mvctrnn": []}
    trace_color_dict = {"mctrnn": [], "mactrnn": [], "mavctrnn": [], "mvctrnn": []}

    # for every past timestep, plot the activations with adjust color values
    assert len(TRACE_COLORS) == len(TRACE_ACTIVATIONS), "trace_colors and trace_acts must have one dict per timestep"

    # 10 steps in the past are considered to be enough to decay under every setting.
    # remove older data, to keep computational growth under control
    if len(TRACE_COLORS) >= 10:
        TRACE_COLORS = TRACE_COLORS[-10:]
        TRACE_ACTIVATIONS = TRACE_ACTIVATIONS[-10:]

    for k, e in reversed(list(enumerate(TRACE_COLORS))):
        # update the colors to slowly fade
        for model in TRACE_COLORS[k].keys():
            # iterate over colors values backwards, so that more past >> more decay
            for color_ind in range(len(TRACE_COLORS[k][model])):
                old_color = TRACE_COLORS[k][model][color_ind]
                # new_alpha = 1 - (j / (j + 1))  # compute decaying alpha value
                new_alpha = TRACE_DECAY_GAMMA ** k
                if new_alpha == 1:
                    new_alpha = 0.75

                new_rgba = list(mpl.colors.to_rgb(old_color))
                new_rgba.append(new_alpha)
                TRACE_COLORS[- (k + 1)][model][color_ind] = new_rgba

    # set_offsets method expects N [x,y] tuples -.- hence the hacky data formatting
    # save the values from the current step in dicts, that will be saved in lists, so that we can access the data later
    for k in range(1300):
        mctrnn_xy.append([models["mctrnn"]["tau_vals"][k], models["mctrnn"]["hidac_vals"][i][k]])
        mctrnn_colors.append(C_PAL["tauplot_scatter"])
        # we use the same color theme as in the 3x4 freqpred animation, thus "neuron_0"

        mactrnn_xy.append([models["mactrnn"]["tau_vals"][k], models["mactrnn"]["hidac_vals"][i][k]])
        mactrnn_colors.append(C_PAL["tauplot_scatter"])

        mavctrnn_xy.append([models["mavctrnn"]["tau_vals"][k], models["mavctrnn"]["hidac_vals"][i][k]])
        mavctrnn_colors.append(C_PAL["tauplot_scatter"])

        mvctrnn_xy.append([models["mvctrnn"]["tau_vals"][k], models["mvctrnn"]["hidac_vals"][i][k]])
        mvctrnn_colors.append(C_PAL["tauplot_scatter"])

    # setup local dict
    trace_act_dict["mctrnn"] = mctrnn_xy
    trace_color_dict["mctrnn"] = mctrnn_colors

    trace_act_dict["mactrnn"] = mactrnn_xy
    trace_color_dict["mactrnn"] = mactrnn_colors

    trace_act_dict["mavctrnn"] = mavctrnn_xy
    trace_color_dict["mavctrnn"] = mavctrnn_colors

    trace_act_dict["mvctrnn"] = mvctrnn_xy
    trace_color_dict["mvctrnn"] = mvctrnn_colors

    # save the local trace dict to global list
    TRACE_ACTIVATIONS.append(trace_act_dict)
    TRACE_COLORS.append(trace_color_dict)

    # finally create flat lists with all (current and trace) data
    all_mctrnn_acts = []
    all_mactrnn_acts = []
    all_mavctrnn_acts = []
    all_mvctrnn_acts = []
    for k in range(len(TRACE_ACTIVATIONS)):
        for model in TRACE_ACTIVATIONS[k]:
            if model == "mctrnn":
                all_mctrnn_acts.extend(TRACE_ACTIVATIONS[k][model])
            elif model == "mactrnn":
                all_mactrnn_acts.extend(TRACE_ACTIVATIONS[k][model])
            elif model == "mavctrnn":
                all_mavctrnn_acts.extend(TRACE_ACTIVATIONS[k][model])
            elif model == "mvctrnn":
                all_mvctrnn_acts.extend(TRACE_ACTIVATIONS[k][model])

    all_mctrnn_colors = []
    all_mactrnn_colors = []
    all_mavctrnn_colors = []
    all_mvctrnn_colors = []
    for k in range(len(TRACE_COLORS)):
        for model in TRACE_COLORS[k]:
            if model == "mctrnn":
                all_mctrnn_colors.extend(TRACE_COLORS[k][model])
            elif model == "mactrnn":
                all_mactrnn_colors.extend(TRACE_COLORS[k][model])
            elif model == "mavctrnn":
                all_mavctrnn_colors.extend(TRACE_COLORS[k][model])
            elif model == "mvctrnn":
                all_mvctrnn_colors.extend(TRACE_COLORS[k][model])

    # set the data
    mctrnn_scatter.set_offsets(all_mctrnn_acts)
    mctrnn_scatter.set_color(np.asarray(all_mctrnn_colors))

    mactrnn_scatter.set_offsets(all_mactrnn_acts)
    mactrnn_scatter.set_color(np.asarray(all_mactrnn_colors))

    mavctrnn_scatter.set_offsets(all_mavctrnn_acts)
    mavctrnn_scatter.set_color(np.asarray(all_mavctrnn_colors))

    mvctrnn_scatter.set_offsets(all_mvctrnn_acts)
    mvctrnn_scatter.set_color(np.asarray(all_mvctrnn_colors))

    # set the color of the current word
    dyn_color_list = [mpl.colors.to_rgb(C_PAL["fig_title"]) for word in words]
    dyn_color_list[i] = mpl.colors.to_rgba(C_PAL["highglight_red"])

    for ind, artist in enumerate(word_arists):
        artist.set_color(dyn_color_list[ind])

    return mctrnn_scatter, mactrnn_scatter, mavctrnn_scatter, mvctrnn_scatter


# setup the writer
ffmpeg = animation.writers['ffmpeg'](fps=FPS, extra_args=["-s", str(RES_WIDTH) + "x" + str(RES_HEIGHT)])

# timesteps = 10
ani = animation.FuncAnimation(fig, update_data, init_func=init, blit=False, frames=timesteps, interval=700)
# ani = animation.FuncAnimation(fig, update_data, init_func=init, blit=False, frames=10, interval=700)
ani.save(filename='animated_ptb_trace.mp4', writer=ffmpeg, dpi=DPI)
