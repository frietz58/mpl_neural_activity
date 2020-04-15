import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import seaborn as sns
from matplotlib.colors import ListedColormap
# from matplotlib import rc
# rc('text', usetex=True)
# plt.rc('text.latex', preamble=r"\usepackage{xcolor}")


# some globals
MAIN_FOLDER = "./ptb"
BATCHID = 19
WORDS_PATH = "ptb_annotation.txt"
TOP_MARIGIN = 40
# BOT_MARGIN = 40
RES_WIDTH = 1920
RES_HEIGHT = 1080
DPI = 100
ART_TEXT_MARGIN = 0.0035  # this controls how much spacing is between the the current word and the surrounding text artists
FPS = 1
NUM_PREV_WORDS = 3
NUM_FOLLOWING_WORDS = 3
pal = sns.cubehelix_palette(100)
ANNOTATION = r"consumers may want to move their telephones a little closer to the tv set <unk> <unk> watching abc 's monday night football can now vote during <unk> for the greatest play in N years from among four or five <unk> <unk>"
# normal fonts
# ANNOTATION_FS = 15
# XLABEL_FS = 10
# YLABEL_FS = 10
# SUBPLOT_TITLE_FS = 13
# FIG_FS = 18
# BOT_MARGIN = 40
# TICKS_FS = 10

# big fonts
ANNOTATION_FS = 15
XLABEL_FS = 13
YLABEL_FS = 13
SUBPLOT_TITLE_FS = 15
FIG_FS = 18
TICKS_FS = 13
BOT_MARGIN = 60

XLIM = 12
SNS_COLORS = ListedColormap(sns.color_palette(sns.cubehelix_palette(100, reverse=True)).as_hex())  # make own cmap from seaborn palette

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

# read the words
words = []
with open(WORDS_PATH) as f:
    for line in f:
        for word in line.split():
            words.append(word)

# setup plot grid (2x2 tiles: MCTRNN, MACTRNN / MAVTRNN, MVTRNN)
fig = plt.figure(figsize=(RES_WIDTH / DPI, RES_HEIGHT / DPI))  # 960 x 500 px per scatter
ax_mctrnn = fig.add_subplot(221)
ax_mactrnn = fig.add_subplot(222)
ax_mvctrnn = fig.add_subplot(223)
ax_mavctrnn = fig.add_subplot(224)

# for later iteration
axes = [ax_mctrnn, ax_mactrnn, ax_mvctrnn, ax_mavctrnn]

# set titles and prepare animated text
fig.suptitle("Penn TreeBank Language Modelling", fontfamily="sans-serif", fontweight="bold", fontsize=FIG_FS)
ax_mctrnn.set_title("CTRNN", fontsize=SUBPLOT_TITLE_FS)
ax_mactrnn.set_title("ACTRNN", fontsize=SUBPLOT_TITLE_FS)
ax_mavctrnn.set_title("AVCTRNN", fontsize=SUBPLOT_TITLE_FS)
ax_mvctrnn.set_title("VCTRNN", fontsize=SUBPLOT_TITLE_FS)

# all the stuff that is the same for all subplots
for axis in axes:
    # axis.set_xlim(0, max(models["mctrnn"]["tau_vals"]))
    axis.set_xlim(0, XLIM)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Hidden unit's timescale 4", fontsize=XLABEL_FS)
    axis.set_ylabel("Hidden activation", fontsize=YLABEL_FS)
    axis.tick_params(labelsize=TICKS_FS)

# those values will mostly be overwritten in the animate function
full_text = plt.figtext(0.5, 0.01, fontsize=ANNOTATION_FS, ha="center", s="placeholder")

# global init of the artists
mctrnn_scatter = ax_mctrnn.scatter([], [])
mactrnn_scatter = ax_mactrnn.scatter([], [])
mavctrnn_scatter = ax_mavctrnn.scatter([], [])
mvctrnn_scatter = ax_mvctrnn.scatter([], [])

# rect "squeezes" all the subplots into the figure area specified by the rect
top_percentage = TOP_MARIGIN / RES_HEIGHT
bot_percentage = BOT_MARGIN / RES_HEIGHT
plt.tight_layout(rect=(0, bot_percentage, 1, 1 - top_percentage))
plt.subplots_adjust(hspace=0.25, wspace=0.2)


# init function needed for blitting and passing references
def init():
    return mctrnn_scatter, mactrnn_scatter, mavctrnn_scatter, mvctrnn_scatter


# the update function
def update_data(i):
    mctrnn_xy = []
    mactrnn_xy = []
    mavctrnn_xy = []
    mvctrnn_xy = []

    mctrnn_colors = []
    mactrnn_colors = []
    mavctrnn_colors = []
    mvctrnn_colors = []

    # set_offsets method expects N [x,y] tuples -.- hence the hacky data formatting
    for j in range(1300):
        mctrnn_xy.append([models["mctrnn"]["tau_vals"][j], models["mctrnn"]["hidac_vals"][i][j]])
        mctrnn_colors.append(models["mctrnn"]["hidac_vals"][i][j])

        mactrnn_xy.append([models["mactrnn"]["tau_vals"][j], models["mactrnn"]["hidac_vals"][i][j]])
        mactrnn_colors.append(models["mactrnn"]["hidac_vals"][i][j])

        mavctrnn_xy.append([models["mavctrnn"]["tau_vals"][j], models["mavctrnn"]["hidac_vals"][i][j]])
        mavctrnn_colors.append(models["mavctrnn"]["hidac_vals"][i][j])

        mvctrnn_xy.append([models["mvctrnn"]["tau_vals"][j], models["mvctrnn"]["hidac_vals"][i][j]])
        mvctrnn_colors.append(models["mvctrnn"]["hidac_vals"][i][j])

    mctrnn_scatter.set_offsets(mctrnn_xy)
    # mctrnn_scatter.set_cmap(SNS_COLORS)  # setting the colormap on init doesn't seem to work :(
    # mctrnn_scatter.set_array(np.asarray(mctrnn_colors))

    mactrnn_scatter.set_offsets(mactrnn_xy)
    # mactrnn_scatter.set_cmap(SNS_COLORS)
    # mactrnn_scatter.set_array(np.asarray(mactrnn_colors))

    mavctrnn_scatter.set_offsets(mavctrnn_xy)
    # mavctrnn_scatter.set_cmap(SNS_COLORS)
    # mavctrnn_scatter.set_array(np.asarray(mavctrnn_colors))

    mvctrnn_scatter.set_offsets(mvctrnn_xy)
    # mvctrnn_scatter.set_cmap(SNS_COLORS)
    # mvctrnn_scatter.set_array(np.asarray(mvctrnn_colors))

    # make the current word bold
    words = ANNOTATION.split(" ")
    curr_word = words[i]
    words[i] = r"$\bf{" + curr_word + "}$"  # replace current with bold
    # words[i] = r"\textcolor{red}{" + curr_word + "}"  # TODO get the color to work. \textcolor{red}{word} doesnt work, even with proper latex backend -.-
    words.insert(16, "\n")  # insert line breaks where we want them
    highlighted_text = " ".join(words)  # make one string again
    full_text.set_text(highlighted_text)

    return mctrnn_scatter, mactrnn_scatter, mavctrnn_scatter, mvctrnn_scatter, full_text


# setup the writer
ffmpeg = animation.writers['ffmpeg'](fps=FPS, extra_args=["-s", str(RES_WIDTH) + "x" + str(RES_HEIGHT)])

ani = animation.FuncAnimation(fig, update_data, init_func=init, blit=False, frames=timesteps, interval=700)
ani.save(filename='animated_ptb.mp4', writer=ffmpeg, dpi=DPI)
