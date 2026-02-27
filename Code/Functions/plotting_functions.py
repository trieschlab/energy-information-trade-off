# plotting functions

# import predefined functions
import Functions.analysis_functions as af

# import packages for simulation and calculation
import numpy as np
from brian2 import *
# brian2 is unstable with python 3.12. In case of errors either use: 
#prefs.codegen.target = "numpy"  # or
import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

# import plotting packages
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import matplotlib.gridspec as gs
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.linalg import eigh
from scipy.interpolate import griddata
import random
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import welch #, iirnotch, filtfilt
from matplotlib.colors import Normalize
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from skimage.measure import marching_cubes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# import video packages
import os
import shutil
import imageio
from copy import deepcopy
from plotly.io import write_image





############################# plotting functions Padamsey #############################

############################ basic plotting functions ############################

# define parameters
"""
params = {
    'figure.figsize' : [8,5],
    'text.usetex' : True, # for final plots True
    'axes.titlesize' : 17,
    'axes.labelsize' : 15, 
    'legend.fontsize': 15,
    'xtick.labelsize' : 15,
    'ytick.labelsize' : 15,
    #'xtick.minor.visible' : True,
    #'ytick.minor.visible' : True,
    #'figure.dpi' : 300,
    'font.family' : 'serif',
    'font.serif'  : 'cm',
    'font.size'   : 15
}
#plt.rcParams.update(params) 
"""


def w_scale_to_w_e_syn(w_scale, N_e=3750):
    # translate w_scale to mean synaptic weight
    # input
    # w_scale is an array of synaptic scaling values
    # N_e is the number of excitatory synapses
    
    # output
    # w_e_syn is an array of the mean excitatory synaptic weights
    
    w_e_syn = (5 * 4 * np.array(w_scale)) / N_e # outputs same shape as input
    
    return w_e_syn  

# We use the following convention: 
# excitatory input: red
# inhibitory input: blue
# postsynaptic activity: gray



############################ analysis plotting functions ############################

def filter_zero_arrays(curves):
    # filter out all zero array
    # input
    # curves are the tuning curves 
    
    # output
    # filtered_curves are the filtered curves
    # num_excluded is the number of excluded trials
    # num_total is the total number of trials
    
    filtered_curves = [x for x in curves if not np.all(x == 0)]
    num_excluded = len(curves) - len(filtered_curves)
    num_total = len(curves)
    
    return filtered_curves, num_excluded, num_total

def filter_zero_values(result_values):
    # filter out all zero values
    # input
    # result_values are the values of results
    
    # output
    # filtered_values are the filtered values
    # num_excluded is the number of excluded trials
    # num_total is the total number of trials
    
    filtered_values = [x for x in result_values if x != 0]
    num_excluded = len(result_values) - len(filtered_values)
    num_total = len(result_values)
    
    return filtered_values, num_excluded, num_total

def filter_zeros_per_column(result_values):
    # filter out all zero values of array
    # input
    # result_values are the values of results of shape (n_trials, n_metrics)
    
    # output
    # filtered_values are the filtered values
    # num_excluded is the number of excluded trials
    # num_total is the total number of trials

    y = np.array(result_values)
    filtered_means = []
    filtered_stds = []
    num_excluded_per_column = []
    num_total = y.shape[0]

    for i in range(y.shape[1]):
        col = y[:, i]
        non_zero_values = col[col != 0]
        filtered_means.append(np.mean(non_zero_values) if non_zero_values.size > 0 else 0)
        filtered_stds.append(np.std(non_zero_values) / np.sqrt(non_zero_values.size) if non_zero_values.size > 0 else 0)
        num_excluded_per_column.append(num_total - non_zero_values.size)
    
    y_mean = np.array(filtered_means)
    y_std = np.array(filtered_stds)
    
    return y_mean, y_std, num_excluded_per_column, num_total

############################ template plotting functions ############################


def plot_template_one_graph(x, y, x_label, y_label, description, color='dimgray', figsize=(8, 5), ax=None, savename=None):
    # plots a one graphs in a plot
    
    # input
    # x is an array or list with the values of the x-axis
    # y is an array or list for the values of the y-axis
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # description is a string for the title of the plot
    # color set the color of the lines
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # main plotting part
    ax.plot(x, y, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(description)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

    
def plot_template_two_graph(x_1, y_1, x_2, y_2, x_label, y_label, label_1, label_2, description, color_1='blue', color_2='orange', y_1_std=None, y_2_std=None, plot_mode='correlation', figsize=(8, 5), ax=None, savename=None):
    # plots a two graphs in one plot
    
    # input
    # x is an array or list with the values of the x-axis
    # y is an array or list for the values of the y-axis
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # label_1, label_2 are the labels of the plotted values
    # description is a string for the title of the plot
    # color_1, color_2 are the colors to be used
    # y_1_std, y_2_std are optional standard deviations/errors
    # plot_mode is the mode of x & y labels
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # main plotting part
    if y_1_std is None: 
        ax.plot(x_1, y_1, label = label_1, color=color_1)
        ax.plot(x_2, y_2, label = label_2, color=color_2)
    if y_1_std is not None:
        ax.errorbar(x_1, y_1, yerr=y_1_std, label=label_1, color=color_1, fmt='-o', capsize=3)
        ax.errorbar(x_2, y_2, yerr=y_2_std, label=label_2, color=color_2, fmt='-o', capsize=3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(description)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)
    
    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def plot_template_three_graph(x_1, y_1, x_2, y_2, labels_1, labels_2, colors, x_label, y_label, description, figsize=(8, 5), ax=None, savename=None):
    # plots three graphs in one plot
    
    # input
    # x is an array or list with the values of the x-axis
    # y is a tuple of array or list for the values of the y-axis
    # label_1, label_2 are the labels of the plotted values
    # color_1, color_2 are the colors to be used
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # description is a string for the title of the plot
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    y_1_a, y_1_b, y_1_c = y_1 #y_1_std, y_1_Ee, y_1_Ei = y_1
    label_1_a, label_1_b, label_1_c = labels_1
    y_2_a, y_2_b, y_2_c = y_2 #y_2_std, y_2_Ee, y_2_Ei = y_2
    label_2_a, label_2_b, label_2_c = labels_2
    color_1, color_2 = colors

    # E_L variation 
    ax.plot(x_1, y_1_a, label=label_1_a, color=color_1, linestyle='-')
    ax.plot(x_1, y_1_b, label=label_1_b, color=color_1, linestyle='--')
    ax.plot(x_1, y_1_c, label=label_1_c, color=color_1, linestyle=':')

    # V_thresh variation
    ax.plot(x_2, y_2_a, label=label_2_a, color=color_2, linestyle='-')
    ax.plot(x_2, y_2_b, label=label_2_b, color=color_2, linestyle='--')
    ax.plot(x_2, y_2_c, label=label_2_c, color=color_2, linestyle=':')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(description)
    ax.legend(frameon=False)

    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
        
def plot_template_n_graph(x_list, y_list, labels, x_label, y_label, title, colors=None, yerr_list=None, figsize=(4, 3), ax=None, savename=None):
    # generic line plot for multiple curves

    # input
    # x_list, y_list are lists of arrays of the data to plot
    # labels is a list of strings
    # colors is a list of colors
    # yerr_list is a list of arrays for errorbars
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename_mode is an optional name to save figure

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    n = len(x_list)
    if colors is None:
        colors = [None] * n
    if yerr_list is None:
        yerr_list = [None] * n

    for x, y, label, color, yerr in zip(x_list, y_list, labels, colors, yerr_list):
        if yerr is None:
            ax.plot(x, y, label=label, color=color)
        else:
            ax.errorbar(x, y, yerr=yerr, label=label, color=color, fmt='-o', capsize=3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False)

    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    if standalone:
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

    return fig, ax

def plot_one_histogram(y_1, x_label, y_label, label_1, description, color='grey', axis_mode='linear', figsize=(6, 5), ax=None, savename=None):
    # plot histogram of membrane voltage fluctuations with and without membrane noise

    # input
    # y_1 is an arrays of data
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # label_1 is the label of the plotted values
    # description is a string for the title of the plot
    # color is the color to use for the plot
    # axis_mode is a string switching the axis mode to log if provided
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # main plotting part
    # handle binning
    if axis_mode == 'log':
        # filter out non-positive values to avoid log errors
        y_1 = y_1[y_1 > 0]
        bins = np.logspace(np.log10(min(y_1)), np.log10(max(y_1)), 20)
        ax.set_xscale('log')
    else:
        bins = 20

    # plot histogram
    counts, bin_edges, _ = ax.hist(y_1, bins=bins, label=label_1, color=color)
    ax.legend(frameon=False)
    ax.set_title(description)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # fix axes ticks
    #ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    if axis_mode == 'log':
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
        ax.xaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def plot_two_histograms(y_1, y_2, x_label, y_label, label_1, label_2, description, color_1='black', color_2='red', bins=np.linspace(-77, -67, 100), density=True, figsize=(6, 5), ax=None, savename=None):
    # plot histogram of membrane voltage fluctuations with and without membrane noise

    # input
    # y_1, y_2 are arrays of data
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # label_1, label_2 are the labels of the plotted values
    # description is a string for the title of the plot
    # color_1, color_2 are the colors to be used
    # bins is an array of bins for the histogram with default np.linspace(-77, -67, 100)
    # density decides if histogram is normalized or not
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    ax.hist(y_1, label=label_1, alpha=0.8, bins=bins, color=color_1, density=density)
    ax.hist(y_2, label=label_2, alpha=0.8, bins=bins, color=color_2, density=density)
    ax.legend(frameon=False)
    ax.set_title(description)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
    
############################ illustrative plotting functions ############################

def illustrative_spike():
    # creates a spike 
    # input

    # output
    # t_plot is the time vector of a spike in ms
    # V_plot is the voltage trace of a spike in mV
    
    # time vector (ms)
    t = np.linspace(0, 20, 2000)
    
    # resting potential
    V_rest = -70
    V_peak = 40
    V_undershoot = -90  # stronger undershoot
    
    # spike timing
    t0 = 10  # onset (ms)
    
    # shape parameters — sharper spike
    rise_time = 0.3    # ms
    fall_time = 1    # ms
    undershoot_width = 4.0  # ms
    
    # initialize voltage
    V = np.ones_like(t) * V_rest
    
    def sharp_spike(t):
        # fast sigmoid rise
        rise = 1 / (1 + np.exp(-(t-t0)/rise_time/2.4))
        # sharp exponential fall after peak
        fall = np.exp(-(t-t0)/fall_time)
        # strong gaussian undershoot
        undershoot = np.exp(-((t-(t0+fall_time*2))/undershoot_width)**2)
        
        # combine
        spike = V_rest \
              + (V_peak - V_rest) * rise * fall \
              - (V_rest - V_undershoot) * undershoot
        return spike
    
    V += sharp_spike(t)
    
    V_plot = V#(np.array(V)+57)
    t_plot = t#np.linspace(0, len(V_plot), len(V_plot))

    return t_plot, V_plot


def plot_illustrative_sliding_variable_gap(t_plot, V_plot, color_1='black', color_2='black', fontsize=9, figsize=(3, 4), ax=None, savename=None):
    # plot variable gap
    
    # input
    # t_plot is list of time values in ms
    # V_plot is list of voltage values in mV
    # color_1, color_2 set the color of the arrows
    # fontsize sets the font size
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(t_plot, V_plot, color='black') # , lw=3
    
    # add Vrest and Vthresh
    v_rest = V_plot[0]
    v_thresh = V_plot[600]
    ax.hlines(v_rest, t_plot[0], t_plot[-500], linestyles='--', color='black') #, linewidth=2
    ax.hlines(v_thresh, t_plot[0], t_plot[-500], linestyles='--', color='black') # , linewidth=2
    
    # add arrows
    arrow_x = t_plot[0] - 5
    Delta_x = 30
    
    """
    # aligned in middle
    ax.annotate('', xy=(arrow_x, v2+8), xytext=(arrow_x, v_rest-8), arrowprops=dict(arrowstyle='<->', lw=3, color='black'))
    ax.annotate('', xy=(arrow_x+2, v2+1), xytext=(arrow_x+2, v_rest-1), arrowprops=dict(arrowstyle='<->', lw=3, color='black'))
    ax.annotate('', xy=(arrow_x+4, v2-2), xytext=(arrow_x+4, v_rest+2), arrowprops=dict(arrowstyle='<->', lw=3, color='black'))
    # aligned in middle
    ax.annotate('', xy=(arrow_x +4 + Delta_x, v2+8), xytext=(arrow_x +4 + Delta_x, v_rest-8), arrowprops=dict(arrowstyle='<->', lw=3, color='black'))
    ax.annotate('', xy=(arrow_x + 2 + Delta_x, v2+1), xytext=(arrow_x + 2 + Delta_x, v_rest-1), arrowprops=dict(arrowstyle='<->', lw=3, color='black'))
    ax.annotate('', xy=(arrow_x + Delta_x, v2-2), xytext=(arrow_x + Delta_x, v_rest+2), arrowprops=dict(arrowstyle='<->', lw=3, color='black'))
    """
    # aligned with threshold
    ax.annotate('', xy=(arrow_x, v_thresh), xytext=(arrow_x, v_thresh-35), arrowprops=dict(arrowstyle='<-', color=color_1)) # , lw=3
    ax.annotate('', xy=(arrow_x+2, v_thresh), xytext=(arrow_x+2, v_thresh-25), arrowprops=dict(arrowstyle='<-', color=color_1)) # , lw=3
    ax.annotate('', xy=(arrow_x+4, v_thresh), xytext=(arrow_x+4, v_thresh-15), arrowprops=dict(arrowstyle='<-', color=color_1)) # , lw=3
    # aligned with resting
    ax.annotate('', xy=(arrow_x + Delta_x, v_rest ), xytext=(arrow_x + Delta_x, v_rest+35), arrowprops=dict(arrowstyle='<-', color=color_2)) #, lw=3
    ax.annotate('', xy=(arrow_x + 2 + Delta_x, v_rest), xytext=(arrow_x + 2 + Delta_x, v_rest+25), arrowprops=dict(arrowstyle='<-', color=color_2)) #, lw=3
    ax.annotate('', xy=(arrow_x + 4 + Delta_x, v_rest), xytext=(arrow_x + 4 + Delta_x, v_rest+15), arrowprops=dict(arrowstyle='<-', color=color_2)) #, lw=3
    
    # style
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    #ax.set_xlim(arrow_x - 1, t[-1])
    ax.set_xlim(arrow_x - 1, arrow_x + Delta_x + 4)
    
    # add labels
    ax.text(t_plot[-1]-4, v_thresh+2, r'$V_\mathrm{th}$', va='center', ha='left')#, fontsize=fontsize)
    ax.text(t_plot[-1]-4, v_rest+4, r'$V_{rest}$ ', va='center', ha='left')#, fontsize=fontsize)
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            #fig.savefig("sliding_variable_gap.png", dpi=300, bbox_inches="tight", pad_inches=0.01, transparent=False)
            print(f"Saved figure to {path}")
        plt.show()

def plot_illustrative_sliding_constant_gap(t_plot, V_plot, color_1='black', color_2='red', fontsize=9, figsize=(3, 4), ax=None, savename=None):
    # plot constant gap
    
    # input
    # t_plot is list of time values in ms
    # V_plot is list of voltage values in mV
    # color_1, color_2 set the color of the arrows
    # fontsize sets the font size
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(t_plot, V_plot, color='black') # , lw=3
    
    # add Vrest and Vthresh
    v_rest = V_plot[0]
    v_thresh = V_plot[600]
    ax.hlines(v_rest, t_plot[0], t_plot[-500], linestyles='--', color='black') # , linewidth=2
    ax.hlines(v_thresh, t_plot[0], t_plot[-500], linestyles='--', color='black') # , linewidth=2
    
    # add arrows
    arrow_x = t_plot[0] - 5
    Delta_x = 30

    ax.annotate('', xy=(arrow_x, v_thresh-4), xytext=(arrow_x, v_rest-6), arrowprops=dict(arrowstyle='<->', color=color_1)) # , lw=3
    ax.annotate('', xy=(arrow_x+2, v_thresh+1), xytext=(arrow_x+2, v_rest-1), arrowprops=dict(arrowstyle='<->', color=color_1)) # , lw=3
    ax.annotate('', xy=(arrow_x+4, v_thresh+6), xytext=(arrow_x+4, v_rest+5), arrowprops=dict(arrowstyle='<->', color=color_1)) # , lw=3
    
    ax.annotate('', xy=(arrow_x+ Delta_x, v_thresh-4), xytext=(arrow_x+ Delta_x, v_rest-6), arrowprops=dict(arrowstyle='<->', color=color_2)) # , lw=3
    ax.annotate('', xy=(arrow_x+2+ Delta_x, v_thresh+1), xytext=(arrow_x+2+ Delta_x, v_rest-1), arrowprops=dict(arrowstyle='<->', color=color_2)) # , lw=3
    ax.annotate('', xy=(arrow_x+4+ Delta_x, v_thresh+6), xytext=(arrow_x+4+ Delta_x, v_rest+5), arrowprops=dict(arrowstyle='<->', color=color_2)) # , lw=3
    
    # style
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    #ax.set_xlim(arrow_x - 1, t[-1])
    ax.set_xlim(arrow_x - 1, arrow_x + Delta_x + 4)
    
    # add labels
    ax.text(t_plot[-1]-4, v_thresh+2, r'$V_\mathrm{th}$', va='center', ha='left')#, fontsize=fontsize)
    ax.text(t_plot[-1]-4, v_rest+4, r'$V_{rest}$', va='center', ha='left')#, fontsize=fontsize)
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            #fig.savefig("sliding_variable_gap.png", dpi=300, bbox_inches="tight", pad_inches=0.01, transparent=False)
            print(f"Saved figure to {path}")
        plt.show()
    

def illustrative_EPSC(figsize=(3, 4), ax=None, savename='EPSC_CTR_FR'):
    # creates CTR & FR EPSC
    # input
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # time vector
    t = np.linspace(0, 200, 1000)  # ms
    
    # EPSC parameters
    A1 = 2      # peak amplitude of black curve (pA)
    A2 = A1*0.65     # peak amplitude of red curve (pA)
    tau_rise = 10   # ms
    tau_decay = 40  # ms
    
    # EPSC function
    def epsc(t, A, tau_rise, tau_decay):
        epsc_val = A * (np.exp(-t/tau_decay) - np.exp(-t/tau_rise))
        # normalize so the peak is at A
        epsc_val /= np.max(epsc_val)
        epsc_val *= A
        return epsc_val
    
    # compute EPSCs
    i_black = -epsc(t, A1, tau_rise, tau_decay)
    i_red   = -epsc(t, A2, tau_rise, tau_decay)
    
    # plot    
    ax.plot(t, i_black, lw=3, color='black')
    ax.plot(t, i_red, lw=3, color='red')
    
    
    # style
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    #plt.xlabel('Time / ms')
    #plt.ylabel('Current / pA')
    #plt.title('EPSC traces')
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            #fig.savefig("sliding_variable_gap.png", dpi=300, bbox_inches="tight", pad_inches=0.01, transparent=False)
            print(f"Saved figure to {path}")
        plt.show()

def illustrative_AP(figsize=(3, 4), ax=None, savename='AP_CTR_FR'):
    # creates CTR & FR EPSC
    # input
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # AdEx model parameters
    C = 200.0       # pF
    gL = 10.0       # nS
    EL = -70.0      # mV
    VT = -50.0      # mV
    DeltaT = 2.0    # mV
    tau_w = 100.0   # ms
    a = 2.0         # nS
    b = 40.0        # pA
    V_reset = -65.0 # mV
    V_spike = 20.0  # mV
    
    # time vector
    dt = 0.1
    T = 50
    n_steps = int(T/dt)
    t = np.linspace(0, T, n_steps)
    
    # input currents
    I_black = 200.0  # pA (subthreshold)
    I_red = 300.0    # pA (suprathreshold)
    
    # function to simulate AdEx
    def adexp(I_inj):
        V = np.zeros(n_steps)
        w = np.zeros(n_steps)
        V[0] = EL
        for i in range(1, n_steps):
            current = I_inj if t[i] <= 30 else 0.0
            dV = ( -gL*(V[i-1]-EL) + gL*DeltaT*np.exp((V[i-1]-VT)/DeltaT) - w[i-1] + current ) / C
            V[i] = V[i-1] + dt * dV
            dw = ( a*(V[i-1]-EL) - w[i-1] ) / tau_w
            w[i] = w[i-1] + dt * dw
            if V[i] >= V_spike:
                V[i-1] = V_spike  # record spike
                V[i] = V_reset
                w[i] += b
        return V
    
    # simulate both cases
    V_black = adexp(I_black)
    V_red = adexp(I_red)
    
    # plot
    fig, ax = plt.subplots(figsize=(3,4))
    
    ax.plot(t, V_black, lw=3, color='black')
    ax.plot(t, V_red, lw=3, color='red')
    
    VT_plot = VT + 2
    ax.hlines(VT_plot, t[0], t[-1], linestyles='--', linewidth=2, color='black')
    ax.text(t[-1] + 2, VT_plot, r'$V_\mathrm{th}$', va='center', ha='left', fontsize=27)
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            #fig.savefig("sliding_variable_gap.png", dpi=300, bbox_inches="tight", pad_inches=0.01, transparent=False)
            print(f"Saved figure to {path}")
        plt.show()
        
    """
    # spike shape from detailed model
    V_spike = [-79.99156213865182, -79.99156196849061, -79.99156179842971, -78.48599112868878, -77.43935004088874, -76.60459051040132, -75.89734986435022, -75.2776094990764, -74.72271755254539, -74.21830885477395, -73.75452776131716, -73.32422020021635, -72.92197357186711, -72.54356362379725, -72.18561421691574, -71.84537659724212, -71.52057978271233, -71.20932538718941, -70.91001137113226, -70.62127528194789, -70.34195101955805, -70.0710352479508, -69.80766084308671, -69.55107558127554, -69.30062480678458, -69.05573718081887, -68.8159128485367, -68.58071353529621, -68.34975418846217, -68.12269589317835, -67.89923983231517, -67.67912211706901, -67.4621093441061, -67.24799476030222, -67.03659495147295, -66.827746972191, -66.62130585503411, -66.41714244557014, -66.215141507269, -66.01520007333433, -65.8172260054317, -65.62113673262976, -65.42685815069872, -65.2343236489048, -65.04347325688197, -64.85425289685892, -64.66661372454139, -64.48051154949488, -64.29590632945977, -64.11276171082073, -63.93104462952319, -63.750724958395686, -63.57177519374006, -63.3941701788396, -63.21788686420413, -63.042904081196625, -62.86920234699438, -62.69676369199544, -62.52557150492106, -62.35561039533133, -62.18686607506734, -62.01932524623947, -61.85297549736616, -61.68780521558964, -61.523803507417846, -61.36096012611835, -61.19926540872055, -61.038710225457386, -60.8792859178582, -60.72098425046461, -60.56379736848419, -60.40771775887992, -60.25273821411444, -60.09885180161816, -59.94605184306932, -59.794331877090926, -59.64368563424459, -59.494107015519376, -59.34559007251728, -59.19812898714771, -59.051718056956574, -58.90635168515592, -58.76202436423588, -58.61873065587014, -58.476465180087686, -58.33522260496578, -58.19499763607135, -58.05578500672532, -57.9175794716294, -57.780375807956545, -57.6441687989818, -57.508953224534324, -57.3747238559832, -57.24147545157389, -57.10920275104347, -56.977900470207715, -56.847563299742944, -56.718185908005914, -56.5897629351737, -56.46228898212913, -56.33575860931207, -56.21016633526512, -56.08550663509567, -55.961773936602285, -55.83896262077881, -55.71706702212761, -55.59608143814222, -55.47600011742785, -55.356817255702666, -55.23852699644964, -55.1211234315887, -55.004600601783046, -54.88895249458017, -54.77417304538602, -54.66025614082624, -54.547195624147314, -54.43498529813066, -54.32361891287419, -54.21309016867241, -54.103392717508726, -53.994520164670256, -53.88646607033367, -53.77922394712588, -53.67278726420818, -53.56714945067739, -53.462303900696085, -53.35824398080877, -53.25496301523342, -53.15245428962351, -53.05071105292634, -52.949726519828964, -52.84949387227913, -52.75000626092795, -52.651256801026996, -52.55323858199282, -52.45594466754565, -52.359368103518825, -52.26350192187691, -52.168339127529066, -52.07387270194276, -51.9800956050335, -51.8870007774155, -51.79458114164377, -51.70282960419476, -51.611739052417434, -51.52130235724425, -51.431512379114594, -51.34236196752249, -51.25384397128662, -51.1659512368317, -51.07867659740488, -50.99201287600253, -50.905952886753276, -50.820489436400344, -50.73561532506671, -50.651323347595344, -50.56760629374391, -50.484456940386096, -50.4018680626387, -50.31983243364061, -50.2383428240792, -50.15739201232381, -50.07697278071955, -49.997077904238964, -49.91770015259724, -49.83883229072124, -49.76046707886843, -49.682597272949245, -49.605215624155726, -49.52831487917862, -49.45188777754826, -49.3759270439746, -49.300425399780835, -49.22537556017676, -49.15077023293602, -49.0766021233979, -49.00286393890809, -48.929548369537905, -48.85664808966157, -48.78415575688557, -48.7120640106223, -48.640365470933396, -48.56905273703951, -48.498118385172106, -48.42755496677672, -48.35735500489769, -48.28751098122212, -48.2180153488385, -48.148860527483485, -48.08003889952376, -48.01154280714779, -47.9433645656322, -47.875496440428684, -47.80793064104343, -47.74065931844256, -47.673674560945294, -47.606968389643654, -47.540532754227996, -47.474359528144575, -47.40844050267862, -47.342767381820394, -47.27733177547064, -47.21212518058727, -47.14713898659241, -47.08236447060616, -47.01779278772455, -46.953414963015476, -46.88922188439849, -46.82520430891317, -46.76135282660642, -46.697657853163356, -46.634109618746294, -46.57069815575027, -46.50741328518815, -46.44424460320725, -46.38118146598027, -46.31821297244754, -46.255327947656625, -46.19251492240711, -46.129762104236434, -46.0670573562845, -46.00438818537722, -45.94174171353114, -45.879104649720176, -45.81646325931961, -45.75380334652592, -45.691110206711905, -45.628368578043776, -45.56556260171965, -45.50267577534929, -45.43969090337574, -45.37659004111045, -45.313354435555716, -45.249964458221555, -45.18639953221672, -45.122638051151014, -45.05865729006593, -44.99443328243935, -44.929940738201914, -44.86515292073201, -44.80004150844356, -44.734576443219375, -44.66872576569164, -44.602455436568945, -44.535729092438736, -44.468507805613, -44.40074980743457, -44.33241017365231, -44.26344046438062, -44.19378831332889, -44.12339695329503, -44.052204669411104, -43.980144163138895, -43.907141797718396, -43.83311673017205, -43.75797991417445, -43.68163288831148, -43.60396633876499, -43.524858397472094, -43.444172556283476, -43.36175512216717, -43.27743212893051, -43.191005519022944, -43.10224838235449, -43.01089896091849, -42.91665301227161, -42.819153966142665, -42.71798001316965, -42.612627108233575, -42.50248598154973, -42.386810586979486, -42.26467384175579, -42.134903984818855, -41.99599072755056, -41.84594228940897, -41.68205943474887, -41.50056231309724, -41.29594142991063, -41.05975424091015, -40.77821072164868, -40.42682726328908, -39.95702380084336, -39.25714141941748, -38.02472077205925, -35.46100769306768, -30.683645652448696, -23.222676375644088, -9.638779658697407, 11.829335791377845, 25.296808684494053, 27.022856988364893, 23.178164966562665, 17.82117849078784, 12.785978341203863, 8.598670738114357, 5.24003195617442, 2.548159640247293, 0.36298601933444496, -1.4411021059745235, -2.95666242686078, -4.251081981239308, -5.373771350308107, -6.361530128824328, -8.317693962665944, -9.866940090589976, -11.200273531376068, -12.375533840374626, -13.428920267454348, -14.387859271167736, -15.274711698274999, -16.108587776365372, -16.90638175936508, -17.683373638648142, -18.45358774544423, -19.22999336338616, -20.024585408985256, -20.848360054500322, -21.71118374059545, -22.621469304138554, -23.585605585491834, -24.607092603512527, -25.68543536783562, -26.814963502036782, -27.983985264812894, -29.17480723057787, -30.365045029085298, -31.53026121864226, -32.64745262305553, -33.69846596827011, -34.67219231231896, -35.564809880199775, -36.37834880447628, -37.118543205115984, -37.7928734868339, -38.40919296361612, -38.9749579819111, -39.496863588628315, -39.98073841479439, -40.43156909513724, -40.85358712595276, -41.25036794706432, -41.62493925748024, -41.979871787406125, -42.31736081950252, -42.63929616728031, -42.94731703707327, -43.24285646643028, -43.527177277126604, -43.80139916256203, -44.06651843170685, -44.32342370935164, -44.5729078262754, -44.81567752242964, -45.052361338375036, -45.283516652459134, -45.50963613871876, -45.731154184744256, -45.9484528191343, -46.16186746201151, -46.37169228423266, -46.57818491882406, -46.78157129534467, -46.98204986707772, -47.17979535603016, -47.374962036336164, -47.56768639754031, -47.758089424461446, -47.94627877165314, -48.13235025917044, -48.31638909584817, -48.498470891819515, -48.6786624140633, -48.8570223940174, -49.03360251593018, -49.20844827311156, -49.3815999445423, -49.55309367186097, -49.722962439383764, -49.8912370207752, -50.0579471190831, -50.22312221714182, -50.38679227349955, -50.548988253004616, -50.70974233923438, -50.86908786512354, -51.027059066392724, -51.18369070171742, -51.339017409431115, -51.493072978022, -51.64588960437072, -51.79749714953193, -51.9479224805651, -52.09718896629473, -52.245316231190955, -52.39231999885013, -52.53821209318259, -52.68300060386345, -52.82669016142656, -52.969282265531604, -53.11077566451303, -53.25116687442907, -53.39045066952086, -53.528620546270645, -53.665669160034575, -53.80158874940247, -53.93637150418611, -54.07000987617626, -54.202496817419416, -54.33382606875132, -54.46399236020047, -54.59299154963194, -54.720820734174964, -54.84747832723709, -54.97296411209559, -55.097279239494796, -55.220426214855586, -55.34240885988088, -55.463232314146865, -55.58290297765289, -55.70142844443842, -55.81881743007514, -55.93507969573075, -56.050225985690176, -56.16426792334149, -56.27721793104772, -56.38908912345569, -56.49989526359997, -56.60965068310106, -56.71837020553905, -56.82606907339994, -56.93276288223498, -57.038467517016066, -57.14319910265118, -57.2469739295654, -57.34980840771838, -57.4517190072065, -57.552722226898375, -57.65283456361968, -57.75207247590337, -57.850452352424576, -57.9479904835534, -58.04470303471931, -58.14060602528952, -58.2357153163554, -58.330046582048936, -58.42361529719179, -58.51643671870849, -58.608525874234736, -58.6998975645342, -58.79056635463243, -58.880546566211564, -58.969852272164104, -59.058497291050124, -59.14649518318067, -59.233859250543176, -59.32060253871567, -59.40673782633395, -59.49227762624726, -59.57723418418283, -59.66161946742284, -59.74544518153384, -59.82872276929566, -59.911463412029185, -59.993678029483085, -60.075377281947816, -60.156571571838526, -60.237271044621366, -60.31748559195441, -60.39722486026306, -60.47649824433875, -60.55531489168262, -60.63368370621279, -60.711613346898154, -60.78911223293289, -60.86618855428324, -60.94285027383751, -61.0191051319961, -61.09496065024587, -61.17042413512327, -61.245502682885416, -61.32020318323099, -61.394532323727915, -61.468496600083924, -61.54210231565403, -61.61535558549973, -61.68826234167288, -61.76082833813011, -61.83305914906983, -61.90496018118495, -61.97653667926418, -62.04779373059599, -62.11873626935394, -62.18936908081373, -62.25969680532413, -62.32972394279525, -62.39945485644505, -62.468893776139566, -62.53804480496253, -62.606911923549724, -62.67549899000782, -62.743809744328395, -62.81184781212728, -62.87961670777387, -62.94711983210313, -63.01436048403612, -63.081341862368085, -63.148067068666606, -63.21453911006499, -63.280760901745154, -63.34673526948088, -63.412464952282434, -63.477952604878794, -63.54320079926279, -63.60821202750315, -63.67298870799473, -63.73753318278521, -63.80184772009499, -63.86593451685216, -63.92979570067953, -63.993433330550964, -64.05684939669442, -64.12004582754446, -64.1830244907692, -64.24578719500401, -64.30833569157589, -64.37067167589248, -64.43279678905812, -64.49471261947339, -64.55642070429597, -64.61792253040181, -64.67921953576865, -64.74031311275681, -64.80120460921425, -64.8618953278456, -64.9223865280142, -64.98267942712012, -65.04277520194637, -65.10267498806176, -65.16237988097251, -65.2218909405229, -65.28120919149522, -65.3403356247223, -65.39927119823325, -65.45801683808166, -65.51657343934396, -65.57494186713491, -65.63312295757325, -65.69111751854349, -65.74892633018118, -65.80655014617149, -65.86398969672416, -65.92124568696288, -65.97831879780631, -66.03520968701098, -66.09191899003716, -66.148447320915, -66.20479527144948, -66.26096341217746, -66.31695229547489, -66.37276245585831, -66.4283944106205, -66.48384866059136, -66.5391256905869, -66.59422597003392, -66.64914995355365, -66.70389808156507, -66.7584707808304, -66.81286846464964, -66.86709153340897, -66.92114037550033, -66.9750153691864, -67.02871688104236, -67.0822452667619, -67.13560087172918, -67.18878403153586, -67.24179507248306, -67.29463431136162, -67.34730205476055, -67.39979860217727, -67.45212424601904, -67.50427927186301, -67.55626395896357, -67.60807858051139, -67.65972340391413, -67.71119869112164, -67.76250469894839, -67.81364167937181, -67.8646098798417, -67.91540954322939, -67.96604090818737, -68.0165042095362, -68.06679967984267, -68.11692754797973, -68.16688803960112, -68.21668137746651, -68.26630778172859, -68.31576747020058, -68.3650606586178, -68.41418755956387, -68.46314838367, -68.51194334063628, -68.56057263921777, -68.60903648733728, -68.65733509236989, -68.70546866120544, -68.75343740038454, -68.80124151623585, -68.84888121501757, -68.8963567030573, -68.9436681868851, -68.99081587325959, -69.03779996905389, -69.08462068157212, -69.13127821891581, -69.17777279050561, -69.22410460628257, -69.27027387695192, -69.31628081414149, -69.36212563053888, -69.40780854000987, -69.45332975769857, -69.49868949966212, -69.54388798251648, -69.58892542485292, -69.63380204709468, -69.67851807148325, -69.72307372213632, -69.76746922517817, -69.81170480871448, -69.85578070286762, -69.89969713981489, -69.94345435381913, -69.98705258126878, -70.03049206071316, -70.07377303287832, -70.11689574046177, -70.15986042828145, -70.202667343319, -70.24531673516287, -70.2878088558167, -70.33014395938717, -70.3723223021794, -70.41434414275727, -70.4562097419805, -70.49791936302958, -70.53947327142936, -70.58087173506662, -70.6221150233072, -70.6632034080645, -70.70413716383878, -70.74491656764226, -70.78554189896795, -70.82601343979991, -70.86633147465473, -70.90649629052223, -70.94650817685084, -70.98636742553232, -71.02607433088617, -71.06562918965011, -71.10503230096445, -71.14428396635476, -71.18338448968208, -71.22233417697724, -71.26113333651847, -71.29978227882859, -71.33828131686897, -71.3766307660482, -71.41483094385251, -71.45288216989482, -71.49078476592717, -71.52853905583466, -71.56614536562574, -71.60360402341823, -71.64091535942084, -71.6780797059111, -71.71509739671824, -71.75196876769708, -71.78869415685574, -71.82527390426938, -71.86170835203038, -71.8979978442102, -71.9341427268644, -71.97014334800747, -72.00600005756344, -72.04171320732911, -72.07728315094164, -72.11271024384673, -72.1479948432674, -72.18313730816976, -72.21813799923163, -72.25299727881061, -72.28771551087439, -72.32229306088658, -72.35673029584247, -72.39102758424978, -72.4251852961171, -72.45920380322369, -72.49308347866888, -72.52682469690629, -72.56042783373634, -72.5938932662892, -72.62722137300263, -72.66041253359622, -72.69346712904286, -72.72638554154162, -72.75916815449081, -72.7918153524239, -72.82432752065766, -72.85670504576193, -72.88894831546132, -72.92105771858192, -72.95303364501095, -72.98487648566116, -73.01658663244548, -73.0481644782774, -73.0796104170185, -73.1109248434427, -73.14210815320345, -73.17316074280222, -73.20408300955842, -73.23487535157808, -73.26553816772426, -73.29607185758708, -73.32647682145458, -73.35675346028435, -73.3869021756475, -73.4169233696538, -73.44681744496158, -73.47658480476011, -73.50622585274252, -73.53574099316437, -73.56513063084188, -73.59439517096656, -73.62353501910893, -73.65255058120528, -73.68144226353876, -73.71021047271705, -73.73885561564991, -73.76737809952684, -73.79577833179411, -73.82405672013249, -73.85221367243466, -73.88024959678243, -73.90816490127243, -73.93595999412128, -73.96363528372144, -73.99119117859553, -74.01862808736442, -74.04594641872008, -74.07314658140055, -74.10022898416636, -74.12719403579446, -74.1540421450582, -74.18077372069774, -74.2073891713962, -74.23388890575676, -74.2602733322812, -74.28654285934866, -74.31269789519489, -74.3387388478916, -74.36466612532627, -74.39048013518254, -74.41618128492118, -74.44176998176097, -74.46724663265996, -74.49261164426261, -74.51786542288421, -74.543008374504, -74.56804090475187, -74.59296341889406, -74.6177763218169, -74.6424800181016, -74.66707491192726, -74.6915614070201, -74.71593990665103, -74.74021081362507, -74.76437453026894, -74.78843145841796, -74.81238199940245, -74.8362265540346, -74.85996552259518, -74.88359930482011, -74.90712829988749, -74.9305529064046, -74.95387352239517, -74.97709054528659, -75.00020437184763, -75.0232153981754, -75.0461240197653, -75.06893063148375, -75.09163562755047, -75.11423940152389, -75.13674234628783, -75.15914485403911, -75.18144731627551, -75.20365012378431, -75.22575366664283, -75.247758334205, -75.2696645150874, -75.29147259715698, -75.31318296751998, -75.3347960125113, -75.35631211768393, -75.37773166779881, -75.39905504681496, -75.42028263787994, -75.44141482332053, -75.46245198463379, -75.48339450247816, -75.50424275666478, -75.52499712614892, -75.54565798902168, -75.56622572250181, -75.58670070291198, -75.60708330566906, -75.62737390528362, -75.64757287535518, -75.6676805885663, -75.68769741667631, -75.70762373051565, -75.72745989998552, -75.74720629410997, -75.76686328094294, -75.78643122757431, -75.80591050012838, -75.82530146376017, -75.84460448265135, -75.86381992000582, -75.88294813804524, -75.90198949800441, -75.92094436012705, -75.93981308366132, -75.95859602685552, -75.97729354695382, -75.99590600019202, -76.01443374179352, -76.03287712596544, -76.05123650589474, -76.0695122337446, -76.08770466065086, -76.10581413667538, -76.12384101085124, -76.14178563118458, -76.159648344647, -76.17742949717048, -76.1951294336433, -76.21274849790647, -76.23028703275041, -76.24774537991193, -76.26512388007134, -76.28242287284972, -76.29964269680632, -76.31678368944016]
    t_spike = np.arange(0,len(V_spike), len(V_spike))
    
    V_spike_rise = [0:250]
    V_spike_spike = [250:400]
    
    plt.plot(V_spike)
    """
        
############################ synaptic input plotting functions ############################


def load_and_plot_tuning_curves_N_e_signal_ratios(model_mode, N_e_signal_ratios, savename_basic):
    # load saved CT & FR results for given model and N_e_signal_ratios & extract tuning curves, and plot normalized & non-normalized versions.
    # input 
    # model_mode is a list of strings or string & decides which model is used for simulation
    # N_e_signal_ratio portion of signalling synapses
    # savename_basic is the name to save the grid as .pkl file
    
    # output 
    # results_LIF, results_AdExp are the results dictionaries
    
    if isinstance(model_mode, str):
        model_mode = [model_mode]  # normalize input type

    for ratio in N_e_signal_ratios:
        percent = int(round(ratio*100))
        base = f"{savename_basic}_{percent}"

        # initialize dicts for both models
        tuning_curves_CTR_LIF, tuning_curves_FR_LIF, tuning_curves_CTR_AdExp, tuning_curves_FR_AdExp = None, None, None, None

        # load model results as available
        if 'LIF' in model_mode:
            name_CTR_LIF = f"N_exc_signal/{base}_results_single_runs_CTR_LIF"
            name_FR_LIF  = f"N_exc_signal/{base}_results_single_runs_FR_LIF"
            res_CTR_LIF  = af.load_data(name_CTR_LIF)
            res_FR_LIF   = af.load_data(name_FR_LIF)
            tuning_curves_CTR_LIF = res_CTR_LIF["tuning_curve"]
            tuning_curves_FR_LIF = res_FR_LIF["tuning_curve"]

        if 'AdExp' in model_mode:
            name_CTR_AdExp = f"N_exc_signal/{base}_results_single_runs_CTR_AdExp"
            name_FR_AdExp  = f"N_exc_signal/{base}_results_single_runs_FR_AdExp"
            res_CTR_AdExp  = af.load_data(name_CTR_AdExp)
            res_FR_AdExp   = af.load_data(name_FR_AdExp)
            tuning_curves_CTR_AdExp = res_CTR_AdExp["tuning_curve"]
            tuning_curves_FR_AdExp = res_FR_AdExp["tuning_curve"]

        # Plot depending on what was loaded
        if 'LIF' in model_mode: 
            plot_tuning_curves(tuning_curves_CTR_LIF, tuning_curves_FR_LIF, "CTR", "FR", normalized=True, color_CTR="black", color_FR="red", mean_over_zeros=False)

        if 'AdExp' in model_mode:
            plot_tuning_curves(tuning_curves_CTR_AdExp, tuning_curves_FR_AdExp, "CTR", "FR", normalized=True, color_CTR="black", color_FR="red", mean_over_zeros=False)


def plot_raster(spike_times_e, spike_times_i, N_e_noise, w_e=False, title_mode=True, orientation_mode=True, colors=['red', 'blue'], figsize=(12, 8), ax=None, savename=None):
    # plots a raster plot of excitatory and inhibitory spikes
    
    # input
    # spike_times_e is a dictionary of spike times for excitatory neurons
    # spike_times_i is a dictionary of spike times for inhibitory neurons
    # title_mode decides if title is plotted or not
    # colors are the colors for color_input_exc, color_input_inh
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    color_input_exc, color_input_inh = colors[0], colors[1]
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # calculate the number of excitatory neurons
    N_e = len(spike_times_e)
    
    if N_e > 1500: 
        markersize = 0.001
    
    else: 
        markersize = 0.1
    
    # make alpha w_e dependent
    if w_e is not False:
        w = np.array(w_e, dtype=float)
        # normalize to [0, 1]
        w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
        # scale to alpha range [0.2, 0.9]
        alphas = 0.2 + 0.77 * w_norm
    else:
        alphas = np.ones(N_e)
    
    # plot excitatory spikes in red
    for neuron_idx, spike_times in spike_times_e.items():
        alpha = alphas[neuron_idx] if w_e is not False else 1.0
        ax.scatter(spike_times/ms/1000, [neuron_idx] * len(spike_times), color=color_input_exc, s=markersize, alpha=alpha) 

    # plot inhibitory spikes in blue (stacked above excitatory)
    offset_inh = N_e  # offset to stack inhibitory neurons above excitatory
    for neuron_idx, spike_times in spike_times_i.items():
        ax.scatter(spike_times/ms/1000, [neuron_idx + offset_inh] * len(spike_times), color=color_input_inh, s=markersize/2) # 1
    
    ax.set_xlabel('time / s')
    ax.set_ylabel('input neuron index')
    if title_mode: 
        ax.set_title('Raster plot of excitatory (red) and inhibitory (blue) spikes')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # labels on the right side at vertical centers
    N_e = len(spike_times_e)
    N_i = len(spike_times_i)
    total_rows = N_e + N_i

    if total_rows > 0:
        # inhibitory label
        if N_i > 0:
            yfrac_inh = (N_e + N_i / 2.5) / total_rows
            ax.text(0.96, yfrac_inh, 'inh.',transform=ax.transAxes, rotation=270, va='center', ha='left', color='black', clip_on=False)

        # excitatory noise / signal labels
        if N_e > 0 and N_e_noise is not None:
            N_noise  = N_e_noise
            N_signal = max(N_e - N_noise, 0)

            # noise center (first N_noise rows)
            if N_noise > 0:
                yfrac_noise = (N_noise / 2) / total_rows
                ax.text(0.96, yfrac_noise, 'noise syn.', transform=ax.transAxes, rotation=270, va='center', ha='left', color='black', clip_on=False)

            # signal center (remaining excitatory rows)
            if N_signal > 0:
                yfrac_signal = (N_noise + N_signal / 2) / total_rows
                ax.text(0.96, yfrac_signal, 'signal syn.', transform=ax.transAxes, rotation=270, va='center', ha='left', color='black', clip_on=False)
    
    if orientation_mode is True: 
        stimulus_windows = [(3000, 4500), (6000, 7500), (9000, 10500), (12000, 13500), (15000, 16500), (18000, 19500), (21000, 22500)]
        orientation_labels = [r'$0^{\circ}$', r'$30^{\circ}$', r'$60^{\circ}$', r'$90^{\circ}$', r'$60^{\circ}$', r'$30^{\circ}$', r'$0^{\circ}$']
        
        # y positions in axes coordinates (1.0 is the top of the axes)
        y_bar = 0.99
        y_text = 1.01
    
        # Use x in data coords, y in axes fraction
        trans = ax.get_xaxis_transform()
    
        for (t_start, t_end), label in zip(stimulus_windows, orientation_labels):
            # convert ms to s to match x-axis
            x_start = t_start / 1000.0
            x_end   = t_end   / 1000.0
            x_center = 0.5 * (x_start + x_end)
    
            # black bar
            ax.plot([x_start, x_end], [y_bar, y_bar], transform=trans, color='black', clip_on=False) # , linewidth=2
    
            # label above bar
            ax.text(x_center, y_text, label, transform=trans, ha='center', va='bottom', color='black', clip_on=False)
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def plot_cumulative_weights(w_e, description, color='blue', figsize=(5, 4), ax=None, savename=None):
    # plot cumulative weihts function
    
    # input
    # w_e are the excitatory weights to be plotted
    # description is a string for the title of the plot
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    # color is the colors to be used
    
    # translate & order synaptic weights
    w_e = w_scale_to_w_e_syn(w_e, N_e=3750)
    sorted_weights = np.sort(w_e) # sort data
    cdf = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights) # calculate cdf
    
    total_strength = np.sum(sorted_weights)
    weighted_cdf = np.cumsum(sorted_weights) / total_strength # calculate weighted cdf
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    ax.plot(sorted_weights, cdf, label="p(w)", color=color)
    ax.plot(sorted_weights, weighted_cdf, label="wp(w)", color=color, ls="--")
    
    half_strength_idx = np.argmax(weighted_cdf >= 0.5) # add a vertical line at 50% strength
    ax.axvline(sorted_weights[half_strength_idx], color='gray', linestyle='--')
    ax.text(sorted_weights[half_strength_idx]*1.9, 0.45, f"50% of $w_{{e,tot}}$ by \n{round((1-cdf[half_strength_idx])*100)}% of syn.")
    ax.set_xlabel("mean exc weight (nS)") # $⟨w_{syn,e}⟩$ 
    ax.set_ylabel("cum. density frac.")#"cum. density fraction"
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(description)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
    
def plot_synapse_weights_and_rates(w_e, r_e, description, ax_bottom_mode=True, ax_left_mode=True, ax_right_mode=True, colors=['green', 'blue'], figsize=(5,4), ax=None, savename=None):
    # plot synaptic weights (green) and firing rates (blue) against synapse number

    # input
    # w_e is an array of synaptic weights
    # r_e is an array of firing rates
    # description is a string for the title of the plot
    # ax_bottom_mode, ax_left_mode, ax_right_mode decide if the axes are plotted or not
    # colors are the colors for w_e & r_e
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    color_w_e, color_r_e = colors[0], colors[1]
    
    # make sure inputs are numpy arrays
    w_e = np.asarray(w_e)
    r_e = np.asarray(r_e)
    
    # translate w_scale to nS
    #w_e = w_e * 5 * 4 * 100 #w_scale_to_w_e_syn(w_e, N_e=3750)
    
    # sort by synapse number (just index order)
    synapse_indices = np.arange(len(w_e))

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    # create plot    
    # plot firing rates (blue, line)
    line_rates, = ax.plot(synapse_indices, r_e, color=color_r_e, label="rate (Hz)")
    ax.tick_params(axis='y', labelcolor=color_r_e)
    if ax_bottom_mode: 
        ax.set_xlabel('synapse number')
    if ax_bottom_mode is False:
        ax.set_xticklabels([])
    if ax_left_mode:
        ax.set_ylabel('exc rates (Hz)', color=color_r_e) 
    if ax_left_mode is False:
        ax.set_yticklabels([])

    # create second y-axis for synaptic weights
    ax2 = ax.twinx()
    ax2.scatter(synapse_indices, w_e, color=color_w_e, s=20, label="weight (nS)")
    ax2.tick_params(axis='y', labelcolor=color_w_e)
    
    if ax_right_mode:
        ax2.set_ylabel('exc weights (nS)', color=color_w_e) # $⟨w_{syn,e}⟩$  normalized synaptic weights
    if ax_right_mode is False:
        ax2.set_yticklabels([])
        
    # combined legend (handles from both axes)
    #handles = [line_rates, sc_weights]
    #labels = [h.get_label() for h in handles]
    #ax.legend(handles, labels, loc="best", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    
    ax.set_title(description)
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
     
############################### experimental fit plotting functions ############################    

def plot_mEPSC_lognormal_fit(mEPSC_avg_norm_CTR, mEPSC_avg_norm_FR, lognormal_function, label_mode="short", scale_factor=1.845, bins=500, color_CTR='black', color_FR='red', color_scaled='blue', title='Lognormal distribution of mEPSC amplitudes', figsize=(6, 5), ax=None, savename=None):
    # plot mEPSC amplitude distributions with lognormal fits
    
    # input
    # mEPSC_avg_norm_CTR is the experimental normalized CTR distribution
    # mEPSC_avg_norm_FR is the experimental normalized FR distribution
    # lognormal_function is the lognormal model function for curve fitting
    # label_mode selects 'short' or 'long' label style
    # N_e is number of sampled synaptic weights
    # scale_factor is the multiplicative scaling for CTR
    # bins is number of histogram bins
    # color_ctr, color_fr, color_scaled set line colors
    # title is the axis title
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    mEPSCs = np.arange(1, len(mEPSC_avg_norm_CTR)+1, 1)
    # fit lognormal model
    popt_CTR, _ = curve_fit(lognormal_function, mEPSCs, mEPSC_avg_norm_CTR, p0=[1,1], maxfev=10000)
    popt_FR,  _ = curve_fit(lognormal_function, mEPSCs, mEPSC_avg_norm_FR,  p0=[1,1], maxfev=10000)

    # generate fitted functions
    #x_fit = np.linspace(1, len(mEPSC_avg_norm_CTR), 100)
    #y_fit_CTR = pf.lognormal(x_fit, *popt_CTR)
    #y_fit_FR = pf.lognormal(x_fit, *popt_FR)

    # create histograms & scale them: align means & normalize
    N_e=10000000
    w_e_CTR = np.random.lognormal(popt_CTR[0], popt_CTR[1], size=N_e)
    w_e_FR  = np.random.lognormal(popt_FR[0],  popt_FR[1],  size=N_e)
    
    # scale FR to make means of CTR & scaled FR matching
    w_e_CTR_scaled = w_e_CTR / scale_factor
    w_e_FR_scaled = w_e_FR * 1.845

    # compute histogram data
    bins_all = np.histogram_bin_edges(np.concatenate((w_e_CTR, w_e_FR)), bins=bins)
    w_e_CTR_density, bins_CTR = np.histogram(w_e_CTR, bins=bins_all, density=True)
    w_e_FR_density, bins_FR = np.histogram(w_e_FR,  bins=bins_all, density=True)
    w_e_FR_scaled_density, bins_FR_scaled = np.histogram(w_e_FR_scaled,  bins=bins_all, density=True)
    w_e_CTR_scaled_density, bins_CTR_scaled = np.histogram(w_e_CTR_scaled,  bins=bins_all, density=True)
    
    centers = 0.5 * (bins_all[1:] + bins_all[:-1])

    # get label
    if label_mode == "long":
        label_CTR = f'CTR weights fit \n ($\\mu_{{CTR}}$={np.mean(w_e_CTR):.2f}, $\\sigma_{{CTR}}$={np.std(w_e_CTR):.2f})'
        label_FR  = f'FR weights fit \n ($\\mu_{{FR}}$={np.mean(w_e_FR):.2f}, $\\sigma_{{FR}}$={np.std(w_e_FR):.2f})'
        label_scaled = f'CTR multiplicatively scaled \n ($\\mu_{{CTR,ms}}$={np.mean(w_e_CTR_scaled):.2f}, $\\sigma_{{CTR,ms}}$={np.std(w_e_CTR_scaled):.2f})'
    if label_mode == "short":
        label_CTR = 'CTR'
        label_FR  = 'FR'
        label_scaled = 'FR mult. scaled'  
    
    # plot histograms
    ax.plot(centers, w_e_CTR_density, '-', label=label_CTR, color=color_CTR)
    ax.plot(centers, w_e_FR_density, '-', label=label_FR, color=color_FR)
    ax.plot(centers, w_e_CTR_scaled_density, '-', label=label_scaled, color=color_scaled)
    #ax.plot(centers, w_e_FR_scaled_density, '-', label=label_scaled, color=color_scaled)

    # plot experimental data (points)
    #ax.plot(mEPSCs, mEPSC_avg_norm_CTR, '-', label = f'mEPSCs CTR', color= 'black', alpha = 0.2)
    #ax.plot(mEPSCs, mEPSC_avg_norm_FR, '-', label = f'mEPSCs FR', color= 'red', alpha = 0.2)
    ax.scatter(mEPSCs, mEPSC_avg_norm_CTR, color=color_CTR, alpha=0.3, label='CTR mEPSCs')
    ax.scatter(mEPSCs, mEPSC_avg_norm_FR,  color=color_FR,  alpha=0.3, label='FR mEPSCs')
    
    ax.set_xlim(-0.1, 20)
    ax.set_xlabel('mEPSC amplitude (pA)')
    ax.set_ylabel('Density')
    
    # set 3 ticks
    ax.locator_params(axis='x', nbins=3)
    ax.locator_params(axis='y', nbins=3)
    
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
    
def plot_raw_and_clean_V_m(baseline_signal, cleaned_signal, T, color='black', figsize=(8, 5), ax=None, savename=None): 
    # plot the voltage trace and a small inset
    # input
    # baseline_signal is an array of the raw and baseline cleaned signal
    # cleaned_signal is an array of the fully cleaned signal
    # T is the total time in ms
    # color is the graph color
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)#
    else:
        fig = ax.get_figure()

    scalebar_time=5
    scalebar_voltage=10
    inset_time=0.5
    inset_voltage=5
    t = np.linspace(0,T,100000)
    
    ax.plot(t, baseline_signal, color=color, alpha=0.3)
    ax.plot(t, cleaned_signal, color=color, alpha=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-30,100])
    for spine in ax.spines.values(): spine.set_visible(False)
    # scale bar
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x0 = xlim[0] + 0.01 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.01 * (ylim[1] - ylim[0])
    ax.hlines(y0, x0, x0 + scalebar_time, colors='k') # , linewidth=2
    ax.vlines(0.8*x0, y0, y0 + scalebar_voltage, colors='k') # , linewidth=2
    ax.text(x0 + scalebar_time/2, y0 - 0.02*(ylim[1]-ylim[0]), str(scalebar_time)+'ms', ha='center', va='top')
    ax.text(x0 - 0.02*(xlim[1]-xlim[0]), y0 + scalebar_voltage/2, str(scalebar_voltage)+'mV', ha='right', va='center', rotation='vertical')
    # inset
    ins = inset_axes(ax, width='50%', height='30%', loc='upper right')
    ins.plot(t[:1000], baseline_signal[:1000], color=color, alpha=0.3)
    ins.plot(t[:1000], cleaned_signal[:1000], color=color, alpha=1.0)
    ins.set_xticks([]); ins.set_yticks([])
    for spine in ins.spines.values(): spine.set_visible(False)
    # inset scale bar
    xlim = ins.get_xlim(); ylim = ins.get_ylim()
    x0 = xlim[0] - 0.01 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.01 * (ylim[1] - ylim[0])
    ins.hlines(y0, x0, x0 + inset_time, colors='k') # , linewidth=2
    ins.vlines(0.8*x0, y0, y0 + inset_voltage, colors='k') # , linewidth=2
    ins.text(x0 + inset_time/2, y0 - 0.1*(ylim[1]-ylim[0]), str(inset_time)+'ms', ha='center', va='top')
    ins.text(x0 - 0.02*(xlim[1]-xlim[0]), y0 + inset_voltage/2, str(inset_voltage)+'mV', ha='right', va='center', rotation='vertical')

    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def plot_power_spectrum(baseline_signal, cleaned_signal, fs, color='black', figsize=(8, 5), ax=None, savename=None): 
    # plot the power spectrum
    # input
    # baseline_signal is an array of the raw and baseline cleaned signal
    # cleaned_signal is an array of the fully cleaned signal
    # color is the graph color
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)#
    else:
        fig = ax.get_figure()

    clean_interp = cleaned_signal.copy()
    nan_mask = np.isnan(clean_interp)
    if np.any(nan_mask):
        idx = np.arange(len(clean_interp))
        clean_interp[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], clean_interp[~nan_mask])
    f_raw, Pxx_raw = welch(baseline_signal, fs=fs, nperseg=1024)
    f_cln, Pxx_cln = welch(clean_interp, fs=fs, nperseg=1024)
    ax.semilogy(f_raw, Pxx_raw, color=color, alpha=0.3)
    ax.semilogy(f_cln, Pxx_cln, color=color, alpha=1.0)
    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('power')
    ax.grid(True, which='both', ls='--', lw=0.5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    #ax_psd.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    for spine in ax.spines.values(): 
        if spine.spine_type in ['top','right']:
            spine.set_visible(False) 

    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()


def plot_V_m_histograms(baseline_signal, cleaned_signal, color='black', figsize=(8, 5), ax=None, savename=None): 
    # plot two histograms
    # input
    # baseline_signal is list of two arrays of the raw and baseline cleaned signal
    # cleaned_signal is an array of the fully cleaned signal
    # color is the graph color
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)#
    else:
        fig = ax.get_figure()
        
    ax.hist(cleaned_signal[~np.isnan(cleaned_signal)], bins=50, density=True, color=color, alpha=1)
    ax.hist(baseline_signal[~np.isnan(baseline_signal)], bins=500, density=True, color=color, alpha=0.3)
    ax.set_xlabel('$V_{m}$ (mV)')
    ax.set_ylabel('probability density')
    ax.set_xlim([-5,5])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    for spine in ax.spines.values(): 
        if spine.spine_type in ['top','right']:
            spine.set_visible(False) 

    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
        
############################### single runs plotting functions ############################    


def shift_V_m(V_m, target_mV=0.0):
    # shift all excerpts so their global mean equals target_mV
    # input
    # V_m is the membrane voltage array in mV with time step dt_ms
    # target_mV is the desired global mean in mV
    
    # output
    # V_m_shifted is a the shifted membrane voltage array in mV with time step dt_ms
    # Delta_V_m is the applied offset in mV

    V_m_mean = np.mean(V_m)
    Delta_V_m = target_mV - V_m_mean
    V_m_shifted = V_m + Delta_V_m 
    #V_m_excerpts_shifted = [V_m_excerpt + Delta_V_m for V_m_excerpt in V_m_excerpts]
    return V_m_shifted, Delta_V_m

def cut_into_excerpts(V_m, n_excerpts=9, start_idx=3000, stop_idx=30000):
    # cut V_m into equal-length excerpts between start_idx and stop_idx
    
    # input
    # V_m is the membrane voltage array in mV with time step dt_ms
    # n_excerpts is the number of equal excerpts to create
    # start_idx is the starting index (default 3000)
    # stop_idx is the exclusive end index (default 30000)
    
    # output
    # V_m_excerpts is a list of the voltage trace excerpts
    # V_m_min is the minimal membrane voltage in mV
    # V_m_max is the maximal membrane voltage in mV
    # V_m_mean is the mean membrane voltage in mV
    
    seg_len = (stop_idx - start_idx) // n_excerpts

    V_m_excerpts = []
    V_m_min_list = []
    V_m_max_list = []
    V_m_mean_list = []
    
    for k in range(n_excerpts):
        a = start_idx + k * seg_len
        b = a + seg_len
        V_m_seg = np.asarray(V_m[a:b], dtype=float)
        V_m_excerpts.append(V_m_seg)
        V_m_min_list.append(np.min(V_m_seg))
        V_m_max_list.append(np.max(V_m_seg))
        V_m_mean_list.append(np.mean(V_m_seg))
        
    V_m_min = np.min(V_m_min_list)
    V_m_max = np.max(V_m_max_list)   
    V_m_mean = np.mean(V_m_mean_list)

    return V_m_excerpts, V_m_min, V_m_max, V_m_mean


def compute_shared_ylim(V_m_min_1, V_m_max_1, V_m_min_2, V_m_max_2):
    # compute shared y-limits for two excerpt sets with symmetric padding
    # input
    # V_m_min_1 is the min membrane voltage of set 1 in mV
    # V_m_max_1 is the min membrane voltage of set 1 in mV
    # V_m_min_2 is the min membrane voltage of set 2 in mV
    # V_m_max_2 is the min membrane voltage of set 2 in mV
    
    # output
    # V_m_min, V_m_max are the shared y-limits in mV
    
    V_m_min = min(V_m_min_1, V_m_min_2)
    V_m_max = max(V_m_max_1, V_m_max_2)
    
    return V_m_min, V_m_max



############################### single runs plotting functions ############################    

def plot_V_m_excerpts(V_m_excerpts, color="blue", ylims=None, figsize=(5, 4), ax=None, savename=None):
    # plot one set of excerpts on a given axis (first/main + faded backgrounds)
    
    # input
    # V_m_excerpts is a list of (time_seg, V_m_seg)
    # color is the color
    # ylims sets the y-limits (ymin, ymax) with None keeps autoscale
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
     
    main_idx = 0
    t_ms = np.arange(0,len(V_m_excerpts[0]),1) * 0.1
    for i, V_m_excerpt in enumerate(V_m_excerpts):
        if i == main_idx:
            ax.plot(t_ms, V_m_excerpt, lw=2.0, alpha=1.0, color=color)
        else:
            ax.plot(t_ms, V_m_excerpt, lw=0.8, alpha=0.35, color=color)

    if ylims is not None:
        ax.set_ylim(ylims)
        ymin = ylims[0]
        ymax = ylims[1]

    if ylims is None:
        ymin, ymax = ax.get_ylim()

    #ax.set_xlabel("time (s)")
    #ax.set_ylabel("$V_m$ (mV)")

    # remove frame/labels
    ax.axis('off')

    # compute limits after plotting (use autoscale if ylims not provided)
    yr = ymax - ymin

    # scalebar anchor (bottom-left-ish inside the panel)
    x0 = -12
    y0 = ymin - 0.001 * yr

    scale_time_ms = 50 # time scalebar length in ms
    scale_time_dt = scale_time_ms # time scalebar length in ms
    scale_V_m_mV = 5 # voltage scalebar length in mV

    # time scalebar
    ax.plot([x0, x0 + scale_time_dt], [y0, y0], color='black', clip_on=False, zorder=5) # , lw=4.0
    ax.text(x0 + scale_time_dt / 2.0, y0 - 0.04 * yr, f'{int(scale_time_ms)} ms', ha='center', va='top', clip_on=False)

    # voltage scalebar
    ax.plot([x0, x0], [y0, y0 + scale_V_m_mV], color='black', clip_on=False, zorder=5) # , lw=4.0
    ax.text(x0 - 2.0, y0 + scale_V_m_mV / 2.0, f'{int(scale_V_m_mV)} mV', ha='right', va='center', rotation=90, clip_on=False) 
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def plot_two_V_m_excerpts(V_m_excerpts_1, V_m_excerpts_2, color_1="blue", color_2="red", ylims=None, figsize=(5, 4), ax=None, savename=None):
    # plot two first-excerpts on a single axis
    
    # input
    # V_m_excerpts_1, V_m_excerpts_2 are lists of excerpts (time_seg, V_m_seg)
    # color_1, color_2 are the colors
    # ylims sets the y-limits (ymin, ymax) with None keeps autoscale
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # extract only the first excerpt from each
    V1 = np.asarray(V_m_excerpts_1[0])
    V2 = np.asarray(V_m_excerpts_2[0])

    # assume dt = 0.1 ms as in original code
    t_ms = np.arange(0, len(V1), 1) * 0.1

    # plot thick main traces
    ax.plot(t_ms, V1, lw=1.5, alpha=0.9, color=color_1)
    ax.plot(t_ms, V2, lw=1.5, alpha=0.9, color=color_2)

    # y-limits
    if ylims is not None:
        ax.set_ylim(ylims)
        ymin, ymax = ylims
    else:
        ymin = min(V1.min(), V2.min())
        ymax = max(V1.max(), V2.max())
        ax.set_ylim(ymin, ymax)

    # remove all axes / frames
    ax.axis('off')

    # scalebar
    yr = ymax - ymin
    x0 = -12
    y0 = ymin - 0.001 * yr

    scale_time_ms = 50   # ms
    scale_V_m_mV  = 5    # mV

    # time scalebar
    ax.plot([x0, x0 + scale_time_ms], [y0, y0], color='black', clip_on=False, zorder=5) # , lw=4.0
    ax.text(x0 + scale_time_ms/2.0, y0 - 0.04*yr, f'{scale_time_ms} ms', ha='center', va='top', clip_on=False)

    # voltage scalebar
    ax.plot([x0, x0], [y0, y0 + scale_V_m_mV], color='black', clip_on=False, zorder=5) # , lw=4.0
    ax.text(x0 - 2.0, y0 + scale_V_m_mV/2.0, f'{scale_V_m_mV} mV', ha='right', va='center', rotation=90, clip_on=False)

    # standalone saving
    if standalone:
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
        plt.close(fig)
    
def plot_ionic_current(t_vec, I_K, I_Na, I_Ca, label_1, label_2, label_3, color_1, color_2, color_3, x_label, y_label, description, figsize=(8, 5), ax=None, savename=None):
    # plots ionic currents
    
    # input
    # t_vec is an array or list with the time values
    # I_K, I_Na, I_Ca are lists or arrays of ionic (potassium, sodium, calcium) currents in nA
    # label_1, label_2, label_3 are the labels of the plotted values
    # color_1, color_2, color_3 are the colors to be used
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # description is a string for the title of the plot
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.stackplot(t_vec, I_Na, I_Ca, labels=[label_2, label_3], colors=[color_2, color_3], alpha=0.8)
    ax.stackplot(t_vec, I_K, labels=[label_1], colors=[color_1], alpha=0.8)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(description)
    ax.legend(frameon=False)

    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
        
def compute_half_max_positions(x, y, minmax_mode=True):
    # compute FWHM positions for given x and y
    
    # input
    # minmax_mode determines whether min & max values of tuning curves are used (True) or 0 and 1 (False)
    
    # output
    # left & right FWHM position
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    peak_idx = np.argmax(y)
    peak_y = y[peak_idx]

    if peak_y <= 0:
        return None, None  # no positive peak -> no FWHM

    # include the peak point of left side
    y_left_side = y[:peak_idx + 1]
    #x_left_side = x[:peak_idx + 1]
    y_min_left = np.min(y_left_side)
    if minmax_mode is False: 
        y_min_left = 0
    half_left = y_min_left + 0.5 * (peak_y - y_min_left)

    x_left = None
    # go from peak downwards until half crossed
    for i in range(peak_idx, 0, -1):
        y1, y2 = y[i], y[i - 1]
        if (y1 - half_left) * (y2 - half_left) <= 0:  # crossing
            x1, x2 = x[i], x[i - 1]
            if y2 != y1:
                frac = (half_left - y1) / (y2 - y1)
                x_left = x1 + frac * (x2 - x1)
            else:
                x_left = x1
            break

    # include the peak point of right side
    y_right_side = y[peak_idx:]
    #x_right_side = x[peak_idx:]
    y_min_right = np.min(y_right_side)
    if minmax_mode is False: 
        y_min_right = 0
    half_right = y_min_right + 0.5 * (peak_y - y_min_right)

    x_right = None
    # go from peak downwards until half crossed
    for i in range(peak_idx, len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if (y1 - half_right) * (y2 - half_right) <= 0:
            x1, x2 = x[i], x[i + 1]
            if y2 != y1:
                frac = (half_right - y1) / (y2 - y1)
                x_right = x1 + frac * (x2 - x1)
            else:
                x_right = x1
            break

    return x_left, x_right

def FWHM_per_noise_level(results_membrane_noise):
    # compute FWHM for the mean tuning curve at each noise level
    # input
    # results_membrane_noise are the results dictionaries
    
    # output 
    # FWHM vector (len = number of noise levels). np.nan if not computable.
    
    noise_level_vec = np.asarray(results_membrane_noise['membrane_noise_vec'], dtype=float)
    tcs_trials = results_membrane_noise['tuning_curve']
    n_noise = len(noise_level_vec)

   
    orientation_list = [-90, -60, -30, 0, 30, 60, 90]
    
    FWHM = np.full(n_noise, np.nan, dtype=float)

    for j in range(n_noise):
        # mean curve across trials for this noise level
        curves_j = [np.asarray(tc_trial[j], dtype=float) for tc_trial in tcs_trials]
        mean_curve = np.nanmean(np.stack(curves_j, axis=0), axis=0)
        mean_curve_normalized = mean_curve/np.max(mean_curve)
     

        x_left, x_right = compute_half_max_positions(orientation_list, mean_curve_normalized, minmax_mode=False)

        if (x_left is not None) and (x_right is not None):
            FWHM[j] = (abs(x_right) + abs(x_left))/2

    return FWHM
    
def plot_tuning_curves(tuning_curves_CTR, tuning_curves_FR, label_CTR, label_FR, normalized=True, color_CTR = 'black', color_FR = 'red', mean_over_zeros=True, mode='tuning_curve', half_width_max=False, minmax_mode=True, show_legend=True, show_xlabel=True, show_ylabel=True, figsize=(2.5, 3), ax=None, savename=None):
    # plot tuning curves CTR vs FR
    
    # input
    # tuning_curves_CTR, tuning_curves_FR are lists or arrays of the control & food-restricted tuning curves
    # label_CTR, label_FR are the labels of the plotted values
    # normalized is the option of normalizing the tuning curves
    # color_CTR, color_FR are the colors of the tuning curves
    # mean_over_zeros is an optional argument which excludes all 0 values before meaning
    # mode decides if tuning_curve labels or CV_ISI_labels are used
    # half_width_max decides if dashed vertical lines at full width half maximum (FWHM) is drawn
    # minmax_mode determines whether min & max values of tuning curves are used (True) or 0 and 1 (False)
    # show_legend, show_xlabel, show_ylabel decides if legend, xlabel & ylabel should be shown or not
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    
    orientation_list = [-90, -60, -30, 0, 30, 60, 90] 
    xticks = ['-90', '', '', '0', '', '', '90'] 
   
    if type(tuning_curves_CTR) == list: # if input is a list of tuning curves for multiple runs
    
        if mean_over_zeros is False:
            tuning_curves_CTR, num_excluded_CTR, num_total_CTR = filter_zero_arrays(tuning_curves_CTR)
            tuning_curves_FR, num_excluded_FR, num_total_FR = filter_zero_arrays(tuning_curves_FR)
            print(f"Excluded {num_excluded_CTR} non-spiking of {num_total_CTR} CTR trials and {num_excluded_FR} non-spiking of {num_total_FR} FR trials.")
            
        mean_tuning_curve_CTR = np.mean([x for x in tuning_curves_CTR if x is not None], axis=0)
        mean_tuning_curve_FR = np.mean([x for x in tuning_curves_FR if x is not None], axis=0)
    
        standard_error_tuning_curve_CTR = np.std([x for x in tuning_curves_CTR if x is not None], axis=0)/np.sqrt(len(tuning_curves_CTR))
        standard_error_tuning_curve_FR = np.std([x for x in tuning_curves_FR if x is not None], axis=0)/np.sqrt(len(tuning_curves_FR))
    
    if type(tuning_curves_CTR) == np.ndarray: # if input is an array of values for a single run
        mean_tuning_curve_CTR = tuning_curves_CTR
        mean_tuning_curve_FR = tuning_curves_FR
    
        standard_error_tuning_curve_CTR = 0 
        standard_error_tuning_curve_FR = 0
    
    if normalized == False: # raw tuning curve
        ax.errorbar(orientation_list, mean_tuning_curve_CTR, standard_error_tuning_curve_CTR, label = label_CTR, color = color_CTR,fmt='-o', capsize=3)
        ax.errorbar(orientation_list, mean_tuning_curve_FR, standard_error_tuning_curve_FR, label = label_FR, color = color_FR,fmt='-o', capsize=3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    
    if normalized == True: # normalized tuning curve
        ax.errorbar(orientation_list, mean_tuning_curve_CTR/max(mean_tuning_curve_CTR), standard_error_tuning_curve_CTR, label = label_CTR, color = color_CTR, fmt='-o', capsize=3)
        ax.errorbar(orientation_list, mean_tuning_curve_FR/max(mean_tuning_curve_FR), standard_error_tuning_curve_FR, label = label_FR, color = color_FR,fmt='-o', capsize=3)
        ax.set_ylim(-0.01,1.05)
        ax.set_yticks([0, 0.5, 1])
    
    if half_width_max:
        # CTR
        FWHM_L_CTR, FWHM_R_CTR = compute_half_max_positions(orientation_list, mean_tuning_curve_CTR, minmax_mode=minmax_mode)
        if FWHM_L_CTR is not None and FWHM_R_CTR is not None:
            ax.axvline(FWHM_L_CTR, linestyle='--', color=color_CTR, alpha=0.7,linewidth=1.0)
            ax.axvline(FWHM_R_CTR, linestyle='--', color=color_CTR, alpha=0.7, linewidth=1.0)

        # FR
        FWHM_L_FR, FWHM_R_FR = compute_half_max_positions(orientation_list, mean_tuning_curve_FR, minmax_mode=minmax_mode)
        if FWHM_L_FR is not None and FWHM_R_FR is not None:
            ax.axvline(FWHM_L_FR, linestyle='--', color=color_FR, alpha=0.7, linewidth=1.0)
            ax.axvline(FWHM_R_FR, linestyle='--', color=color_FR, alpha=0.7, linewidth=1.0)

    if show_xlabel is True: 
        ax.set_xlabel('Distance from \n Preferred ($\circ$)') 
    
    if show_ylabel is True: 
        if mode == 'tuning_curve' and normalized == False: 
            ax.set_ylabel('Spike Rate (Hz)') 
        if mode == 'tuning_curve' and normalized == True: 
            ax.set_ylabel('Normalized Spike Rate') 
        if mode == 'CV_ISI_tuning_curve' and normalized == False: 
            ax.set_ylabel('CV of ISIs') 
        if mode == 'CV_ISI_tuning_curve' and normalized == True: 
            ax.set_ylabel('Normalized CV of ISIs') 
        
    if show_legend is True: 
        ax.legend(frameon=False) 
    ax.set_xlim(-99,+99)
    ax.set_xticks(orientation_list, xticks)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
    
def plot_multiple_tuning_curves(tuning_curves_CTR_LIF_0_noise_multiplicative, tuning_curves_FR_LIF_0_noise_multiplicative, tuning_curves_FR_LIF_1_noise_multiplicative, tuning_curves_FR_AdExp_1_noise_multiplicative, tuning_curves_FR_AdExp_1_noise_nonmultiplicative, label_CTR, label_FR_LIF_0_noise_multiplicative, label_FR_LIF_1_noise_multiplicative, label_FR_AdExp_1_noise_multiplicative, label_FR_AdExp_1_noise_nonmultiplicative, normalized=True, color_CTR = 'black', color_FR_LIF_0_noise_multiplicative='orange', color_FR_LIF_1_noise_multiplicative='purple', color_FR_AdExp_1_noise_multiplicative='coral', color_FR_AdExp_1_noise_nonmultiplicative = 'red', savename=None):
    # plot mutiple tuning curves CTR vs FR with LIF vs AdExp & no-noise vs noise & multiplicative synaptic scaling vs non-multiplicative synaptic scaling 
    
    # input
    # tuning_curves_CTR, tuning_curves_FR are lists or arrays of the control & food-restricted tuning curves
    # label_1, label_2 are the labels of the plotted values
    # normalized is the option of normalizing the tuning curves
    # color_CTR, color_FR are the colors of the tuning curves
    # mean_over_zeros is an optional argument which excludes all 0 values before meaning
    # mode decides if tuning_curve labels or CV_ISI_labels are used
    # savename is an optional name to save figure
    
    orientation_list = [-90, -60, -30, 0, 30, 60, 90] 
    xticks = ['-90', '', '', '0', '', '', '90'] 
   
    mean_tuning_curve_CTR_LIF_0_noise_multiplicative = np.mean([x for x in tuning_curves_CTR_LIF_0_noise_multiplicative if x is not None], axis=0)
    mean_tuning_curve_FR_LIF_0_noise_multiplicative = np.mean([x for x in tuning_curves_FR_LIF_0_noise_multiplicative if x is not None], axis=0)
    mean_tuning_curve_FR_LIF_1_noise_multiplicative = np.mean([x for x in tuning_curves_FR_LIF_1_noise_multiplicative if x is not None], axis=0)
    mean_tuning_curve_FR_AdExp_1_noise_multiplicative = np.mean([x for x in tuning_curves_FR_AdExp_1_noise_multiplicative if x is not None], axis=0)
    mean_tuning_curve_FR_AdExp_1_noise_nonmultiplicative = np.mean([x for x in tuning_curves_FR_AdExp_1_noise_nonmultiplicative if x is not None], axis=0)
    
    standard_error_tuning_curve_CTR_LIF_0_noise_multiplicative = np.std([x for x in tuning_curves_CTR_LIF_0_noise_multiplicative if x is not None], axis=0)/np.sqrt(len(tuning_curves_CTR_LIF_0_noise_multiplicative))
    standard_error_tuning_curve_FR_LIF_0_noise_multiplicative = np.std([x for x in tuning_curves_FR_LIF_0_noise_multiplicative if x is not None], axis=0)/np.sqrt(len(tuning_curves_FR_LIF_0_noise_multiplicative))
    standard_error_tuning_curve_FR_LIF_1_noise_multiplicative = np.std([x for x in tuning_curves_FR_LIF_1_noise_multiplicative if x is not None], axis=0)/np.sqrt(len(tuning_curves_FR_LIF_1_noise_multiplicative))
    standard_error_tuning_curve_FR_AdExp_1_noise_multiplicative = np.std([x for x in tuning_curves_FR_AdExp_1_noise_multiplicative if x is not None], axis=0)/np.sqrt(len(tuning_curves_FR_AdExp_1_noise_multiplicative))
    standard_error_tuning_curve_FR_AdExp_1_noise_nonmultiplicative = np.std([x for x in tuning_curves_FR_AdExp_1_noise_nonmultiplicative if x is not None], axis=0)/np.sqrt(len(tuning_curves_FR_AdExp_1_noise_nonmultiplicative))

    
    plt.figure(figsize=[4,5])
    
    plt.errorbar(orientation_list, mean_tuning_curve_CTR_LIF_0_noise_multiplicative/max(mean_tuning_curve_CTR_LIF_0_noise_multiplicative), standard_error_tuning_curve_CTR_LIF_0_noise_multiplicative, label = label_CTR, color = color_CTR, fmt='-o', capsize=3)
    plt.errorbar(orientation_list, mean_tuning_curve_FR_LIF_0_noise_multiplicative/max(mean_tuning_curve_FR_LIF_0_noise_multiplicative), standard_error_tuning_curve_FR_LIF_0_noise_multiplicative, label = label_FR_LIF_0_noise_multiplicative, color = color_FR_LIF_0_noise_multiplicative,fmt='-o', capsize=3)
    plt.errorbar(orientation_list, mean_tuning_curve_FR_LIF_1_noise_multiplicative/max(mean_tuning_curve_FR_LIF_1_noise_multiplicative), standard_error_tuning_curve_FR_LIF_1_noise_multiplicative, label = label_FR_LIF_1_noise_multiplicative, color = color_FR_LIF_1_noise_multiplicative,fmt='-o', capsize=3)
    plt.errorbar(orientation_list, mean_tuning_curve_FR_AdExp_1_noise_multiplicative/max(mean_tuning_curve_FR_AdExp_1_noise_multiplicative), standard_error_tuning_curve_FR_AdExp_1_noise_multiplicative, label = label_FR_AdExp_1_noise_multiplicative, color = color_FR_AdExp_1_noise_multiplicative,fmt='-o', capsize=3)
    plt.errorbar(orientation_list, mean_tuning_curve_FR_AdExp_1_noise_nonmultiplicative/max(mean_tuning_curve_FR_AdExp_1_noise_nonmultiplicative), standard_error_tuning_curve_FR_AdExp_1_noise_nonmultiplicative, label = label_FR_AdExp_1_noise_nonmultiplicative, color = color_FR_AdExp_1_noise_nonmultiplicative,fmt='-o', capsize=3)
    plt.ylim(-0.01,1.05)
        
    plt.xlabel('Distance from Preferred ($\circ$)') 
    plt.ylabel('Normalized Spike Rate / Hz') 
            
    plt.legend() 
    plt.xlim(-99,+99)
    plt.xticks(orientation_list, xticks)
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig('../Figures/' + str(savename)+'.pdf')
        #plt.savefig('../Figures/tuning_curves_CTR_FR.pdf')
    plt.show()


def plot_phase_plane_Vw(I_syn_mean_CTR, I_syn_mean_FR, g_L_CTR, g_L_FR, E_L_CTR, E_L_FR, V_thresh, Delta_T, a, tau_w, model_name='AdExp', V_min=-80.0, V_max=-40.0, n_V=400, colors=['black', 'red'], figsize=(5,4), ax=None, savename=None):
    # plot phase plane in (V_m, w_ad) for CTR & FR for a given IF-type model
    # input
    # I_syn_mean_CTR is the mean synaptic current for CTR in nA
    # I_syn_mean_FR is the mean synaptic current for FR in nA
    # g_L_CTR is the leak conductance for CTR in nS
    # g_L_FR is the leak conductance for FR in nS
    # E_L_CTR is the leak reversal potential for CTR in mV
    # E_L_FR is the leak reversal potential for FR in mV
    # V_thresh is the spike threshold in mV
    # Delta_T is the slope factor in mV (set to 0 for models without exponential term)
    # a is the subthreshold adaptation in nS (set to 0 for non-adapting models)
    # tau_w is the adaptation time constant in ms (only relevant if a != 0)
    # model_name is the name of the model for the figure title
    # V_min is the minimum membrane potential in the plotted range in mV
    # V_max is the maximum membrane potential in the plotted range in mV
    # n_V is the number of points in the V_m grid
    # colors are the colors for color_CTR, color_FR
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    color_CTR, color_FR = colors[0], colors[1]
    
    # voltage range (mV)
    V_m = np.linspace(V_min, V_max, n_V)

    # define V_T as in the AdExp simulation
    # V_T = V_thresh - 2.5 * Delta_T if Delta_T > 0, else V_T = V_thresh
    if Delta_T > 0:
        V_T = V_thresh - 2.5 * Delta_T
    else:
        V_T = V_thresh

    # V_m-nullcline: dw/dt free, dV_m/dt = 0 -> w_ad = leak_term + exp_term + I_syn
    # units:
    # g_L in nS, V_m and E_L in mV, Delta_T in mV --> g_L * (E_L - V_m) is in pA -> divide by 1000 to get nA
    def V_m_nullcline(V_m, I_syn, g_L, E_L):
        leak_term = g_L * (E_L - V_m) / 1000.0  # nA
        if Delta_T > 0:
            exp_term = g_L * Delta_T * np.exp((V_m - V_T) / Delta_T) / 1000.0  # nA
        else:
            exp_term = 0.0
        V_m_nullcline = leak_term + exp_term + I_syn  # nA
        return V_m_nullcline

    # w_ad-nullcline: dV_m/dt free, dw_ad/dt = 0 -> w_ad = a * (V_m - E_L)
    # a in nS, V_m - E_L in mV, result in pA -> divide by 1000 for nA
    def w_ad_nullcline(V_m, a, E_L):
        w_ad_nullcline = a * (V_m - E_L) / 1000.0  # nA
        return w_ad_nullcline

    # CTR V_m-nullcline
    V_m_nullcline_CTR = V_m_nullcline(V_m, I_syn_mean_CTR, g_L_CTR, E_L_CTR)
    # FR V_m-nullcline
    V_m_nullcline_FR = V_m_nullcline(V_m, I_syn_mean_FR, g_L_FR, E_L_FR)

    # w_ad (adaptation) nullclines (if a != 0)
    if a != 0:
        w_ad_nullcline_CTR = w_ad_nullcline(V_m, a, E_L_CTR)
        w_ad_nullcline_FR = w_ad_nullcline(V_m, a, E_L_FR)
    else:
        w_ad_nullcline_CTR = None
        w_ad_nullcline_FR = None

    # plotting
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    # V_m-nullclines
    ax.plot(V_m, V_m_nullcline_CTR, label='V-nullcline CTR', color=color_CTR, linestyle='-', linewidth=2)
    ax.plot(V_m, V_m_nullcline_FR,  label='V-nullcline FR',  color=color_FR,   linestyle='-', linewidth=2)

    # w_ad-nullclines if present (adapting models)
    distance_labels = []

    if a != 0: # 2D: LIF+ad, AdExp with distance being the separation of V-nullcline and w-nullcline
        ax.plot(V_m, w_ad_nullcline_CTR, label='w-nullcline CTR', color=color_CTR, linestyle='--')
        ax.plot(V_m, w_ad_nullcline_FR,  label='w-nullcline FR',  color=color_FR,   linestyle='--')
    
        # CTR distance
        idx_min_CTR = np.argmin(V_m_nullcline_CTR)
        Vstar_CTR = V_m[idx_min_CTR]
        wV_CTR = V_m_nullcline_CTR[idx_min_CTR]
        ww_CTR = w_ad_nullcline_CTR[idx_min_CTR]
        dist_CTR = np.abs(wV_CTR - ww_CTR)
    
        if Vstar_CTR < V_thresh:
            ax.plot([Vstar_CTR, Vstar_CTR], [ww_CTR, wV_CTR], color=color_CTR, linestyle=':', linewidth=1)
            distance_labels.append(f'distance CTR = {dist_CTR:.3f} nA')
    
        # FR distance
        idx_min_FR = np.argmin(V_m_nullcline_FR)
        Vstar_FR = V_m[idx_min_FR]
        wV_FR = V_m_nullcline_FR[idx_min_FR]
        ww_FR = w_ad_nullcline_FR[idx_min_FR]
        dist_FR = np.abs(wV_FR - ww_FR)
    
        if Vstar_FR < V_thresh:
            ax.plot([Vstar_FR, Vstar_FR], [ww_FR, wV_FR], color=color_FR, linestyle=':', linewidth=1)
            distance_labels.append(f'distance FR = {dist_FR:.3f} nA')
    
    else: # 1D: LIF, LIF+exp with distance being the vertical distance from V-nullcline to w_ad = 0
        ax.axhline(0.0, color='gray', linestyle=':', label='w_ad = 0 (no adap.)')
    
        # CTR robustness = |w_ad(V_nullcline_min)| = |net current|
        idx_min_CTR = np.argmin(V_m_nullcline_CTR)
        Vstar_CTR = V_m[idx_min_CTR]
        wV_CTR = V_m_nullcline_CTR[idx_min_CTR]   # distance to 0
        dist_CTR = np.abs(wV_CTR)
    
        if Vstar_CTR < V_thresh:
            ax.plot([Vstar_CTR, Vstar_CTR], [0, wV_CTR], color=color_CTR, linestyle=':', linewidth=1)
            distance_labels.append(f'distance CTR = {dist_CTR:.3f} nA')
    
        # FR robustness
        idx_min_FR = np.argmin(V_m_nullcline_FR)
        Vstar_FR = V_m[idx_min_FR]
        wV_FR = V_m_nullcline_FR[idx_min_FR]
        dist_FR = np.abs(wV_FR)
    
        if Vstar_FR < V_thresh:
            ax.plot([Vstar_FR, Vstar_FR], [0, wV_FR], color=color_FR, linestyle=':', linewidth=1)
            distance_labels.append(f'distance FR = {dist_FR:.3f} nA')

    # mark threshold
    ax.axvline(V_thresh, color='k', linestyle=':', linewidth=1)
    ax.set_xlim(V_min, V_max)
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max)
    ax.text(V_thresh, y_max * 0.99, r'$V_{\mathrm{thresh}}$', rotation=90, va='top', ha='right')

    ax.set_xlabel(r'$V_m$ (mV)')
    ax.set_ylabel(r'$w_{\mathrm{ad}}$ (nA)')
    """
    title = f'Phase plane ($V_m$ – $w_{{\mathrm{{ad}}}}$) for {model_name}'
    if distance_labels:
        title += '\n' + ', '.join(distance_labels)
    """
    title = f'{model_name}'
    ax.set_title(title)

    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/phase_plane_CTR_FR_{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()


    
def plot_membrane_noise_effect(results_membrane_noise_CTR, results_membrane_noise_FR, metric, error_bars=False, mean_over_zeros=True, description_mode=True, plot_mode='correlation', half_width_max=True, colors=['black', 'red'], figsize=(4, 3), ax=None, savename_mode=False):
    # plot comparison between the effect of membrane noise on CTR & FR 

    # input
    # results_membrane_noise_CTR & results_membrane_noise_FR are the CTR & FR results dictionaries
    # metric is the quantity to plot
    # error_bars is an optional argument to plot errorbars
    # mean_over_zeros is an optional argument which excludes all 0 values before meaning
    # description_mode is an option to plot description or not
    # plot_mode is the mode of x & y labels
    # half_width_max decides if dashed vertical lines at full width half maximum (FWHM) is drawn
    # colors are the colors for color_CTR & color_FR
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename_mode is an optional name to save figure
    
    color_CTR, color_FR = colors[0], colors[1]
    
    y_1_std, y_2_std = None, None   

    x_1 = results_membrane_noise_CTR['membrane_noise_vec']
    x_2 = results_membrane_noise_FR['membrane_noise_vec']
    y_1 = results_membrane_noise_CTR[f'{metric}']
    y_2 = results_membrane_noise_FR[f'{metric}']
    
    if metric == 'E_tot':
        y_1 = np.array(results_membrane_noise_CTR[f'{metric}'])/1e9
        y_2 = np.array(results_membrane_noise_FR[f'{metric}'])/1e9
    
    if mean_over_zeros is False:
        y_1_mean, y_1_std, num_excluded_CTR, num_total_CTR = filter_zeros_per_column(y_1)
        y_2_mean, y_2_std, num_excluded_FR, num_total_FR = filter_zeros_per_column(y_2)
        print(f"Excluded {num_excluded_CTR} non-spiking of {num_total_CTR} CTR trials and {num_excluded_FR} non-spiking of {num_total_FR} FR trials.")
    
    else:
        y_1_mean = np.mean(y_1, axis=0)
        y_2_mean = np.mean(y_2, axis=0)
        
        if error_bars is True:
            y_1_std = np.std(y_1, axis=0) / np.sqrt(len(y_1))
            y_2_std = np.std(y_2, axis=0) / np.sqrt(len(y_2))
    
    x_label = 'Membrane noise level $\sigma$ (mV/ms)'
    y_label = value_key_text_plot(metric, plot_mode=plot_mode) 
    label_1 = 'CTR OSI'
    label_2 = 'FR OSI'
    #label_1 = 'Moving gap $V_{rest} - V_{thresh} = 22$ mV'
    description=""
    if description_mode is True: 
        description = rf'Comparison of {y_label} of CTR and FR for different noise levels'
    
    savename = None
    if savename_mode is True: 
        savename = f'membrane_noise_level_{metric}'
    
    if error_bars is True:
        plot_template_two_graph(x_1, y_1_mean, x_2, y_2_mean, x_label, y_label, label_1, label_2, description, color_1 = color_CTR, color_2 = color_FR, y_1_std=y_1_std, y_2_std = y_2_std, figsize=figsize, ax=ax, savename = savename)
    
    if error_bars is False:
        plot_template_two_graph(x_1, y_1_mean, x_2, y_2_mean, x_label, y_label, label_1, label_2, description, color_1 = color_CTR, color_2 = color_FR, figsize=figsize, ax=ax, savename = savename)
        
    # add FWHM on the right and dashed lines
       
    if half_width_max:
        ax_left = ax if ax is not None else plt.gca()
        ax_right = ax_left.twinx()

        # compute mean FWHM per noise level from tuning curves
        FWHM_CTR = FWHM_per_noise_level(results_membrane_noise_CTR)
        FWHM_FR  = FWHM_per_noise_level(results_membrane_noise_FR)

        ax_right.plot(x_1, FWHM_CTR, linestyle='--', color=color_CTR, label='CTR FWHM')
        ax_right.plot(x_2, FWHM_FR,  linestyle='--', color=color_FR,  label='FR FWHM')

        ax_right.set_ylabel('FWHM (°)')
        ax_right.invert_yaxis()

        # merge legends from both axes
        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_left.legend(h1 + h2, l1 + l2, frameon=False)
        
        ax.spines['right'].set_visible(True)
        #ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.set_xlim(-0.01,4.5)
        

def plot_V_gap_variable(results_multiple_runs_E_L, results_multiple_runs_V_thresh, metric, mean_over_zeros=True, description_mode=False, plot_mode='correlation', colors=['darkcyan', 'darkviolet'], figsize=(8, 5), ax=None, savename_mode=False):
    # plot comparison between V_rest (E_L) & V_thresh changes

    # input
    # results_multiple_runs_E_L, results_multiple_runs_V_thresh are the results dictionaries
    # metric is the quantity to plot
    # mean_over_zeros is an optional argument which excludes all 0 values before meaning
    # description_mode is an option to plot description or not
    # plot_mode is the mode of x & y labels
    # colors are the colors for color_V_rest_var, color_V_thresh_var
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename_mode is an optional name to save figure
    
    color_V_rest_var, color_V_thresh_var = colors[0], colors[1]
    
    x_1 = results_multiple_runs_E_L['V_gap_absolute_list'][0]
    x_2 = results_multiple_runs_V_thresh['V_gap_absolute_list'][0]
    y_1 = results_multiple_runs_E_L[f'{metric}']
    y_2 = results_multiple_runs_V_thresh[f'{metric}']
    
    if metric == 'E_tot': 
        y_1 = np.asarray(y_1)/1e9
        y_2 = np.asarray(y_2)/1e9
    
    if mean_over_zeros is False:
        y_1_mean, y_1_std, num_excluded_E_L, num_total_E_L = filter_zeros_per_column(y_1)
        y_2_mean, y_2_std, num_excluded_V_thresh, num_total_V_thresh = filter_zeros_per_column(y_2)
        print(f"[{metric}] Excluded E_L trials per point: {num_excluded_E_L} of {num_total_E_L}")
        print(f"[{metric}] Excluded V_thresh trials per point: {num_excluded_V_thresh} of {num_total_V_thresh}")
    else:
        y_1_mean = np.mean(y_1, axis=0)
        y_2_mean = np.mean(y_2, axis=0)
    
    x_label = 'Gap between $V_{rest}$ and $V_{thresh}$ (mV)'
    y_label = value_key_text_plot(metric, plot_mode=plot_mode)
    label_1 = 'Increased $V_{rest}$,\n$V_{thresh} = -50$ mV'
    label_2 = 'Decreased $V_{thresh}$,\n$V_{rest} = -72$ mV'
    description=""
    if description_mode is True: 
        description = rf'Comparison of {y_label} for varying gap between $V_{{rest}}$ and $V_{{thresh}}$'
    
    savename = None
    if savename_mode is True: 
        savename = f'V_gap__variable_{metric}'
    
    plot_template_two_graph(x_1, y_1_mean, x_2, y_2_mean, x_label, y_label, label_1, label_2, description, color_1 = color_V_rest_var, color_2 = color_V_thresh_var, plot_mode=plot_mode, figsize=figsize, ax=ax, savename = savename)


def plot_V_gap_variable_E_e_E_i(results_multiple_runs_E_L, results_multiple_runs_V_thresh, metrics_plot, plot_mode='correlation', colors=['darkcyan', 'darkviolet'], figsize=(8, 5), ax=None, savename=None):
    # plot comparison between V_rest (E_L) & V_thresh changes
    
    # input
    # results_multiple_runs_E_L, results_multiple_runs_V_thresh are the results dictionaries
    # metrics_plot describes the metrics to plot
    # plot_mode is the mode of x & y labels
    # colors are the colors for color_V_rest_var, color_V_thresh_var
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    color_V_rest_var, color_V_thresh_var = colors[0], colors[1]
    
    for metric in metrics_plot:
        x_1 = results_multiple_runs_E_L['V_gap_absolute_list']
        y_1 = list(map(list, zip(*results_multiple_runs_E_L[metric])))

        x_2 = results_multiple_runs_V_thresh['V_gap_absolute_list']
        y_2 = list(map(list, zip(*results_multiple_runs_V_thresh[metric])))

        x_label = 'Gap between $V_{rest}$ and $V_{thresh}$ / mV'
        y_label = value_key_text_plot(metric, plot_mode=plot_mode)
        description = rf'Comparison of {y_label} for varying gap between $V_{{rest}}$ and $V_{{thresh}}$'
        
        if savename is not None:
            savename = f'E_e_E_i_V_gap_variable_{metric}.png'
        else:
            savename = None

        y_1 = (y_1[0], y_1[1], y_1[2])
        y_2 = (y_2[0], y_2[1], y_2[2])
        labels_1 = ('E_L increase (standard)', 'E_L increase (E_e +10 mV)', 'E_L increase (E_i -10 mV)')
        labels_2 = ('V_thresh decrease (standard)', 'V_thresh decrease (E_e +10 mV)', 'V_thresh decrease (E_i -10 mV)')
        colors = (color_V_rest_var, color_V_thresh_var)

        # plot with all 3 curves
        plot_template_three_graph(x_1, y_1, x_2, y_2, labels_1, labels_2, colors, x_label, y_label, description, figsize, ax=ax, savename=savename)
        


def plot_V_gap_constant(results_multiple_runs_V_gap, metrics_plot, mean_over_zeros=True, description_mode=True, plot_mode='correlation', figsize=(8, 5), ax=None, savename_mode=False):
    # plot comparison between V_rest (E_L) & V_thresh changes

    # input
    # results_multiple_runs_V_gap are the results dictionaries
    # metrics_plot is the list of quantities to plot
    # mean_over_zeros is an optional argument which excludes all 0 values before meaning
    # description_mode is an option to plot description or not
    # plot_mode is the mode of x & y labels
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename_mode is an optional name to save figure
    
    
    for metric in metrics_plot:
        x_1 = results_multiple_runs_V_gap['V_rests_list'][0]
        y_1 = results_multiple_runs_V_gap[f'{metric}']
        
        if mean_over_zeros is False:
            y_1_mean, y_1_std, num_excluded_trials, num_total_trials = filter_zeros_per_column(y_1)
            print(f"[{metric}] Excluded trials per V_rest: {num_excluded_trials} of {num_total_trials}")
        else:
            y_1_mean = np.mean(y_1, axis=0)
        x_label = '$V_{rest}$ with equi distant $V_{thresh}$ (mV)'
        y_label = value_key_text_plot(metric, plot_mode=plot_mode) 
        #label_1 = 'Moving gap $V_{rest} - V_{thresh} = 22$ mV'
        description=""
        if description_mode is True: 
            description = rf'Comparison of {y_label} for constant gap of 22 mV between $V_{{rest}}$ and $V_{{thresh}}$'
        
        savename = None
        if savename_mode is True: 
            savename = f'V_gap_constant_{metric}'
        
        plot_template_one_graph(x_1, y_1_mean, x_label, y_label, description, figsize=figsize, ax=ax, savename=savename)
        
def plot_CTR_FR_V_gap_constant(results_multiple_CTR_FR_runs_V_gap, metrics_plot, CTR_FR=False, shade_intersection=False, error_bars=False, mean_over_zeros=True, description_mode=True, plot_mode='correlation', colors=['black', 'red'], figsize=(8, 5), ax=None, savename_mode=False):
    # plot comparison between V_rest (E_L) & V_thresh changes

    # input
    # results_multiple_runs_V_gap are the results dictionaries
    # metrics_plot is the list of quantities to plot
    # CTR_FR is an optional argument to plot the experimental CTR & FR values
    # shade_intersection is an optional argument to shade CTR & FR optimal regimes
    # error_bars is an optional argument to plot errorbars
    # mean_over_zeros is an optional argument which excludes all 0 values before meaning
    # description_mode is an option to plot description or not
    # plot_mode is the mode of x & y labels
    # colors are the colors for color_CTR, color_FR
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename_mode is an optional name to save figure
    
    color_CTR, color_FR = colors[0], colors[1]
    
    results_multiple_runs_V_gap_CTR = results_multiple_CTR_FR_runs_V_gap[0]
    results_multiple_runs_V_gap_FR = results_multiple_CTR_FR_runs_V_gap[1]
    
    y_1_std, y_2_std = None, None
    
    for metric in metrics_plot:
        x_1 = results_multiple_runs_V_gap_CTR['V_rests_list'][0]
        x_2 = results_multiple_runs_V_gap_FR['V_rests_list'][0]
        y_1 = results_multiple_runs_V_gap_CTR[f'{metric}']
        y_2 = results_multiple_runs_V_gap_FR[f'{metric}']
        
        if metric == 'E_tot': 
            y_1 = np.asarray(y_1)/1e9
            y_2 = np.asarray(y_2)/1e9
        
        if mean_over_zeros is False:
            y_1_mean, y_1_std, num_excluded_CTR, num_total_CTR = filter_zeros_per_column(y_1)
            y_2_mean, y_2_std, num_excluded_FR, num_total_FR = filter_zeros_per_column(y_2)
            if not error_bars:
                y_1_std, y_2_std = None, None
            
            # filter out all means with less then 20 digits 
            mask_CTR = (np.array(num_total_CTR) - np.array(num_excluded_CTR) >= 20)
            mask_FR  = (np.array(num_total_FR)  - np.array(num_excluded_FR)  >= 20)
            
            y_1_mean[~mask_CTR] = 0
            if y_1_std is not None:
                y_1_std[~mask_CTR] = 0
            
            
            y_2_mean[~mask_FR]  = 0
            if y_2_std is not None:
                y_2_std[~mask_FR]  = 0
    
            print(f"[{metric}] Excluded CTR trials per V_rest: {num_excluded_CTR} of {num_total_CTR}")
            print(f"[{metric}] Excluded FR trials per V_rest: {num_excluded_FR} of {num_total_FR}")
        else:
            y_1_mean = np.mean(y_1, axis=0)
            y_2_mean = np.mean(y_2, axis=0)
            
            if error_bars is True:
                y_1_std = np.std(y_1, axis=0) / np.sqrt(len(y_1))
                y_2_std = np.std(y_2, axis=0) / np.sqrt(len(y_2))
                
        x_label = '$V_{rest}$ with equi distant $V_{thresh}$ (mV)'
        y_label = value_key_text_plot(metric, plot_mode=plot_mode) 
        label_1 = 'CTR'
        label_2 = 'FR'
        description=""
        if description_mode is True: 
            description = rf'Comparison of {y_label} for constant gap between $V_{{rest}}$ and $V_{{thresh}}$ for CTR vs FR'
        
        savename = None
        if savename_mode is True: 
            savename = f'V_gap_constant_CTR_FR_{metric}'
        
        if CTR_FR is False: 
            if y_1_std is None: 
                plot_template_two_graph(x_1, y_1_mean, x_2, y_2_mean, x_label, y_label, label_1, label_2, description, color_1=color_CTR, color_2=color_FR, figsize=figsize, ax=ax, savename=savename)

            if y_1_std is not None:
                plot_template_two_graph(x_1, y_1_mean, x_2, y_2_mean, x_label, y_label, label_1, label_2, description, color_1=color_CTR, color_2=color_FR, y_1_std = y_1_std, y_2_std = y_2_std, figsize=figsize, ax=ax, savename=savename)
            
        if CTR_FR is True: 
            
            standalone = ax is None
            fig = None
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            
            if y_1_std is None: 
                ax.plot(x_1, y_1_mean, label = label_1, color=color_CTR)
                ax.plot(x_2, y_2_mean, label = label_2, color=color_FR)
            if y_1_std is not None:
                ax.errorbar(x_1, y_1_mean, yerr=y_1_std, label=label_1, color=color_CTR, fmt='-o', capsize=3)
                ax.errorbar(x_2, y_2_mean, yerr=y_2_std, label=label_2, color=color_FR, fmt='-o', capsize=3)
                
            ax.axvline(x=-72, color=color_CTR, linestyle='--', linewidth=1.5, label='CTR exp')
            ax.axvline(x=-66, color=color_FR, linestyle='--', linewidth=1.5, label='FR exp')

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if metric == 'OSI':
                max_OSI_CTR = np.max(y_1_mean)
                max_OSI_FR = np.max(y_2_mean)
                
                ax.axhline(y=max_OSI_CTR, color=color_CTR, linestyle=':', linewidth=1.2)
                ax.axhline(y=max_OSI_FR, color=color_FR, linestyle=':', linewidth=1.2)

            if description_mode is True: 
                ax.set_title(description)
            ax.legend(frameon=True)
            
            if shade_intersection == True: 
                x = x_1 # x_1 == x_2
                y_diff = y_1_mean - y_2_mean
                
                # find where the sign of the difference changes
                sign_changes = np.where(np.diff(np.sign(y_diff)))[0]
                
                # Linearly interpolate to get more precise intersection points
                intersections = []
                for i in sign_changes:
                    x0, x1 = x[i], x[i + 1]
                    y0, y1 = y_diff[i], y_diff[i + 1]
                    x_intersect = x0 - y0 * (x1 - x0) / (y1 - y0)
                    intersections.append(x_intersect)
                
                x_split = intersections[0]  # The x-value where the two curves intersect

                # shade x < x_split with color_1
                ax.axvspan(min(x_1), x_split, color=color_CTR, alpha=0.2)
                
                # shade x > x_split with color_2
                ax.axvspan(x_split, max(x_1), color=color_FR, alpha=0.2)
            
            # fix axes ticks
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            
            # save & plot standalone figures
            if standalone is True:       
                if savename is not None:
                    path = f"../Figures/{savename}.pdf"
                    fig.savefig(path, bbox_inches="tight")
                    print(f"Saved figure to {path}")
                plt.show()
                
############################ adaptation plotting functions ############################




############################ correlation plotting functions ############################

def load_data_correlations(results):
    # load data for correlation plots

    # input 
    # results is the dictionary of saved values

    # output
    # R_m, E_L, w_scale, r_post, CV_V_m, I_syn_e, OSI, OSI_per_energy, E_tot, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy are the masked correlation values

    
    # initialize lists
    R_m, E_L, w_scale = [], [], []
    E_tot, r_post, V_m, CV_V_m, I_syn_e, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy, MI_new, MI_new_per_energy, MICE_new, MICE_new_per_energy, TE_new, TE_new_per_energy, TECE_new, TECE_new_per_energy = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    # extract values from the dictionary
    for key in results:
        R_m.append(float(key.split('_')[1]))
        E_L.append(float(key.split('_')[3]))
        w_scale.append(float(key.split('_')[5]))
        
        V_m.append(results[key]['V_m'])
        I_syn_e.append(results[key]['I_syn_e'])
        #I_syn_i.append(results[key]['I_syn_i'])
        r_post.append(results[key]['r_post'])
        E_tot.append(results[key]['E_tot'])
        CV_V_m.append(results[key]['CV_V_m'])
        OSI.append(results[key]['OSI'])
        OSI_per_energy.append(results[key]['OSI_per_energy'])
        #CV_ISI_tuning_curve.append(results[key]['CV_ISI_tuning_curve'])
        #CV_ISI_tuning_curve_per_energy.append(results[key]['CV_ISI_tuning_curve_per_energy'])
        MI_tuning_curve.append(results[key]['MI_tuning_curve'])
        MI_tuning_curve_per_energy.append(results[key]['MI_tuning_curve_per_energy'])
        MICE_tuning_curve.append(results[key]['MICE_tuning_curve'])
        MICE_tuning_curve_per_energy.append(results[key]['MICE_tuning_curve_per_energy'])
        MI_post.append(None)#results[key]['MI_post'])
        MI_post_per_energy.append(None)#results[key]['MI_post_per_energy'])
        MICE_post.append(None)#results[key]['MICE_post'])
        MICE_post_per_energy.append(None)#results[key]['MICE_post_per_energy'])
        CV_ISI.append(results[key]['CV_ISI'])
        CV_ISI_per_energy.append(results[key]['CV_ISI_per_energy'])
        MI.append(results[key]['MI'])
        MI_per_energy.append(results[key]['MI_per_energy'])
        TE.append(results[key]['TE'])
        TE_per_energy.append(results[key]['TE_per_energy'])
        MICE.append(results[key]['MICE'])
        MICE_per_energy.append(results[key]['MICE_per_energy'])
        TECE.append(results[key]['TECE'])
        TECE_per_energy.append(results[key]['TECE_per_energy'])
        MI_new.append(None)#results[key]['MI_new'])
        MI_new_per_energy.append(None)#results[key]['MI_new_per_energy'])
        TE_new.append(None)#results[key]['TE_new'])
        TE_new_per_energy.append(None)#results[key]['TE_new_per_energy'])
        MICE_new.append(None)#results[key]['MICE_new'])
        MICE_new_per_energy.append(None)#results[key]['MICE_new_per_energy'])
        TECE_new.append(None)#results[key]['TECE_new'])
        TECE_new_per_energy.append(None)#results[key]['TECE_new_per_energy'])
        

    # filter out all 0.0 values & convert lists to numpy arrays for that
    mask = np.array(r_post) != 0.0
    
    R_m = np.array(R_m)[mask]
    E_L = np.array(E_L)[mask]
    w_scale = np.array(w_scale)[mask]
    
    V_m = np.array(V_m)[mask]
    I_syn_e = np.array(I_syn_e)[mask]
    #I_syn_i = np.array(I_syn_i)[mask]
    r_post = np.array(r_post)[mask]
    E_tot = np.array(E_tot)[mask]
    CV_V_m = np.array(CV_V_m)[mask]
    
    OSI = np.array(OSI)[mask]
    OSI_per_energy = np.array(OSI_per_energy)[mask]
    MI_tuning_curve = np.array(MI_tuning_curve)[mask]
    MI_tuning_curve_per_energy = np.array(MI_tuning_curve_per_energy)[mask]
    MICE_tuning_curve = np.array(MICE_tuning_curve)[mask]
    MICE_tuning_curve_per_energy = np.array(MICE_tuning_curve_per_energy)[mask]
    MI_post = np.array(MI_post)[mask]
    MI_post_per_energy = np.array(MI_post_per_energy)[mask]
    MICE_post = np.array(MICE_post)[mask]
    MICE_post_per_energy = np.array(MICE_post_per_energy)[mask]
    CV_ISI = np.array(CV_ISI)[mask]
    CV_ISI_per_energy = np.array(CV_ISI_per_energy)[mask]
    MI = np.array(MI)[mask]
    MI_per_energy = np.array(MI_per_energy)[mask]
    TE = np.array(TE)[mask]
    TE_per_energy = np.array(TE_per_energy)[mask]
    MICE = np.array(MICE)[mask]
    MICE_per_energy = np.array(MICE_per_energy)[mask]
    TECE = np.array(TECE)[mask]
    TECE_per_energy = np.array(TECE_per_energy)[mask]
    MI_new = np.array(MI_new)[mask]
    MI_new_per_energy = np.array(MI_new_per_energy)[mask]
    TE_new = np.array(TE_new)[mask]
    TE_new_per_energy = np.array(TE_new_per_energy)[mask]
    MICE_new = np.array(MICE_new)[mask]
    MICE_new_per_energy = np.array(MICE_new_per_energy)[mask]
    TECE_new = np.array(TECE_new)[mask]
    TECE_new_per_energy = np.array(TECE_new_per_energy)[mask]
    
    return R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy, MI_new, MI_new_per_energy, MICE_new, MICE_new_per_energy, TE_new, TE_new_per_energy, TECE_new, TECE_new_per_energy

def linear_func(x, m, c):
    # define linear fitting function

    # input
    # x is the variable
    # m, c are fitting parameters

    # output
    # fitting function
    
    return m * x + c

def sqrt_func(x, a, b):
    # define square root fitting function

    # input
    # x is the variable
    # a, b are fitting parameters

    # output
    # fitting function
    
    return a * np.sqrt(x) + b
    
def log_func(x, a, b, c):
    # define logarithmic fitting function

    # input
    # x is the variable
    # a, b, c are fitting parameters

    # output
    # fitting function
    
    return a * np.log(b * x + c)

def lognormal(x, mu, sigma):
    # define lognormal fitting function
    
    # input
    # x is the variable
    # mu, sigma are fitting parameters

    # output
    # fitting function
    
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))


def rate_information_func(x, I_max, a, n):
    # define exponential fitting function like Harris et al. 2015 in Fig. 5C

    # input
    # x is the variable
    # I_max, a, n are fitting parameters

    # output
    # fitting function
    
    return I_max * (1 - np.exp(-a * x**n))

def linear_rate_information_func(x, I_max, a, n, m, c):
    # define linear exponential fitting function like Harris et al. 2015 in Fig. 5C

    # input
    # x is the variable
    # I_max, a, n, m, c are fitting parameters

    # output
    # fitting function
    
    return rate_information_func(x, I_max, a, n) / (m * x + c)

def sqrt_rate_information_func(x, I_max, a, n, m, c):
    # define sqrt exponential fitting function similar to Harris et al. 2015 in Fig. 5C

    # input
    # x is the variable
    # I_max, a, n, m, c are fitting parameters

    # output
    # fitting function
    
    return rate_information_func(x, I_max, a, n) / (m * np.sqrt(x) + c)


def linear_log_func(x, a, b, c, d):
    # define linear logarithmic fitting function

    # input
    # x is the variable
    # a, b, c, d are fitting parameters

    # output
    # fitting function
    
    return np.log(c * x + d) / (a * x + b) 

def piecewise_linear_exponential_func(x, k, x_max, max_I_per_energy):
    # define piecewise linear exponential fitting function

    # input
    # x is the variable
    # k, max_r, max_MI_per_energy are fitting parameters

    # output
    # fitting function
    
    return np.piecewise(x, [x < x_max, x >= x_max], 
                        [lambda x: max_I_per_energy * x / x_max, 
                         lambda x: max_I_per_energy * np.exp(-(x - x_max) / k)])

def get_w_scale_value_data(results, value, R_m_init, E_L_init):
    # get the highlight points for a desired value

    # input
    # results is the dictionary of results
    # value is the desired value to extract the data for 
    # R_m_init is the desired R_m value
    # E_L_init is the desired E_L value

    # output
    # w_scale_list is the list of extracted w_scale values
    # values_list is the list of extracted values
    
    # extract ranges from keys & find cloest value for given target values
    R_m_range, E_L_range, w_scale_range = af.extract_ranges_from_keys(results)
    R_m = min(R_m_range, key=lambda x: abs(x - R_m_init))
    E_L = min(E_L_range, key=lambda x: abs(x - E_L_init))

    # filter results for desired R_m_init & E_L_init
    filtered_results = {k: v for k, v in results.items() if f"Rm_{R_m}_EL_{E_L}" in k}
    w_scale_list = []
    values_list = []

    # get all w_scale points with respective desired values
    for key, data in filtered_results.items():
        w_scale = key.split('_')[5]
        values = data[value]
        w_scale_list.append(w_scale)
        values_list.append(values)
    return np.array(w_scale_list), np.array(values_list)

def value_key_text_plot(value_key, plot_mode):
    # get string for title or axis label for desired value_key

    # input
    # value_key is the quantity to display color coded
    # plot_mode decides whether label for grid or for correlation is returned

    # output
    # value_key_text is the respective text of the quantity to display color coded

    if plot_mode == 'grid': 
        if value_key == 'r_post': 
            value_key_text = 'r<sub>post</sub><br>(Hz)'
        elif value_key == 'E_tot': 
            value_key_text = 'E<sub>tot</sub><br>(10<sup>9</sup> ATP/s)'
        elif value_key == 'OSI': 
            value_key_text = 'OSI'
        elif value_key == 'OSI_per_energy': 
            value_key_text = 'OSI/E<sub>tot</sub><br>(10<sup>-9</sup>s/ATP)'
        elif value_key == 'MI_tuning_curve': 
            value_key_text = 'MI<sub>tc</sub>(bits)'
        elif value_key == 'MI_tuning_curve_per_energy': 
            value_key_text = 'MI<sub>tc</sub>/E<sub>tot</sub><br>(bits/(10<sup>9</sup>ATP/s))'
        elif value_key == 'MICE_tuning_curve': 
            value_key_text = 'MICE<sub>tc</sub>(bits/Hz)'
        elif value_key == 'MICE_tuning_curve_per_energy': 
            value_key_text = 'MI coding-efficiency <sub>tc</sub>/E<sub>tot</sub><br>(s<sup>2</sup>bits/(10<sup>9</sup>))'
        elif value_key == 'MI_post': 
            value_key_text = 'MI<sub>post</sub>(bits)'
        elif value_key == 'MI_post_per_energy': 
            value_key_text = 'MI<sub>post</sub>/E<sub>tot</sub><br>(bits/(10<sup>9</sup>ATP/s))'
        elif value_key == 'MICE_post': 
            value_key_text = 'MI coding-efficiency<sub>post</sub>(bits/Hz)'
        elif value_key == 'MICE_post_per_energy': 
            value_key_text = 'MI coding-efficiency<sub>post</sub>/E<sub>tot</sub><br>(s<sup>2</sup>bits/(10<sup>9</sup>))'
        elif value_key == 'CV_V_m': 
            value_key_text = 'CV<sub>V<sub>m</sub></sub>'
        elif value_key == 'CV_ISI': 
            value_key_text = 'CV<sub>ISI</sub>'
        elif value_key == 'CV_ISI_per_energy': 
            value_key_text = 'CV<sub>ISI</sub>/E<sub>tot</sub><br>(10<sup>-9</sup>s/ATP)'
        elif value_key == 'MI': 
            value_key_text = 'MI (bits)'
        elif value_key == 'MI_per_energy': 
            value_key_text = 'MI/E<sub>tot</sub><br>(bits/(10<sup>9</sup>ATP/s))'
        elif value_key == 'TE': 
            value_key_text = 'TE (bits)'
        elif value_key == 'TE_per_energy': 
            value_key_text = 'TE/E<sub>tot</sub><br>(bits/(10<sup>9</sup>ATP/s))'
        elif value_key == 'MICE': 
            value_key_text = 'MI coding-efficiency (bits/Hz)'
        elif value_key == 'MICE_per_energy': 
            value_key_text = 'MI coding-efficiency/E<sub>tot</sub><br>(s<sup>2</sup>bits/(10<sup>9</sup>))'
        elif value_key == 'TECE': 
            value_key_text = 'TE coding-efficiency (bits/Hz)'
        elif value_key == 'TECE_per_energy': 
            value_key_text = 'TE coding-efficiency/E<sub>tot</sub><br>(s<sup>2</sup>bits/(10<sup>9</sup>))'
        else: 
            value_key_text = f'{value_key}'
    
    if plot_mode == 'correlation': 
        if value_key == 'r_post': 
            value_key_text = 'Mean firing rate $r_{post}$ (Hz)'
        elif value_key == 'E_tot': 
            value_key_text = 'Total energy $E_{tot}$ ($10^{9}$ ATP/s)'
        elif value_key == 'OSI': 
            value_key_text = 'OSI'
        elif value_key == 'OSI_per_energy': 
            value_key_text = 'OSI per energy ($10^{-9}$s/ATP)'
        elif value_key == 'V_m': 
            value_key_text = 'Membrane voltage $V_{m} (mV)$'
        elif value_key == 'CV_V_m': 
            value_key_text = 'CV of membrane voltage $CV_{V_{m}}$'
        elif value_key == 'CV_ISI': 
            value_key_text = 'CV of ISI $CV_{ISI}$'
        elif value_key == 'CV_ISI_per_energy': 
            value_key_text = 'CV of ISI per energy ($10^{-9}$s/ATP)'
        elif value_key == 'MI_tuning_curve': 
            value_key_text = 'Mutual information based on tc $MI_{tc}$ (bits)'
        elif value_key == 'MI_tuning_curve_per_energy': 
            value_key_text = 'Mutual information based on tc per energy (bits/($10^{9}$ATP/s))'
        elif value_key == 'MICE_tuning_curve': 
            value_key_text = 'MI coding-efficiency based on tc (bits/Hz)'
        elif value_key == 'MICE_tuning_curve_per_energy': 
            value_key_text = 'MI coding-efficiency based on tc per energy (s$^{2}$bits/($10^{9}$ATP))'
        elif value_key == 'MI_post': 
            value_key_text = 'Mutual information based on spike times $MI_{post}$ (bits)'
        elif value_key == 'MI_post_per_energy': 
            value_key_text = 'Mutual information based on spike times per energy (bits/($10^{9}$ATP/s))'
        elif value_key == 'MICE_post': 
            value_key_text = 'MI coding-efficiency based on spike times (bits/Hz)'
        elif value_key == 'MICE_post_per_energy': 
            value_key_text = 'MI coding-efficiency based on spike times per energy (s$^{2}$bits/($10^{9}$ATP))'
        elif value_key == 'MI': 
            value_key_text = 'Mutual information $MI$ (bits)'
        elif value_key == 'MI_per_energy': 
            value_key_text = 'MI per energy (bits/($10^{9}$ATP/s))'
        elif value_key == 'TE': 
            value_key_text = 'Transfer entropy $TE$ (bits)'
        elif value_key == 'TE_per_energy': 
            value_key_text = 'TE per energy (bits/($10^{9}$ATP/s))'
        elif value_key == 'MICE': 
            value_key_text = 'MI coding-efficiency (bits/Hz)'
        elif value_key == 'MICE_per_energy': 
            value_key_text = 'MI coding-efficiency per energy (s$^{2}$bits/($10^{9}$ATP))'
        elif value_key == 'TECE': 
            value_key_text = 'TE coding-efficiency (bits/Hz)'
        elif value_key == 'TECE_per_energy': 
            value_key_text = 'TE coding-efficiency per energy (s$^{2}$bits/($10^{9}$ATP))'
        elif value_key == 'I_syn_e': 
            value_key_text = 'Excitatory synpatic current $I_{syn,e}$ (nA)'
        elif value_key == 'R_m': 
            value_key_text = 'Membrane resistance $R_{m}$ (MOhm)'
        elif value_key == 'E_L': 
            value_key_text = 'Resting potential $V_{rest}$ (mV)'
        elif value_key == 'w_scale': 
            value_key_text = 'synaptic weights'
        else: 
            value_key_text = f'{value_key}'
            
    if plot_mode == 'correlation_short': 
        if value_key == 'r_post': 
            value_key_text = '$r_{post}$ (Hz)'
        elif value_key == 'E_tot': 
            value_key_text = '$E_{tot}$ ($10^{9}$ ATP/s)'
        elif value_key == 'OSI': 
            value_key_text = 'OSI'
        elif value_key == 'OSI_per_energy': 
            value_key_text = 'OSI/$E_{tot}$ (1/($10^{9}$ATP/s))'
        elif value_key == 'V_m': 
            value_key_text = '$V_{m} (mV)$'
        elif value_key == 'CV_V_m': 
            value_key_text = '$CV_{V_{m}}$'
        elif value_key == 'CV_ISI': 
            value_key_text = '$CV_{ISI}$'
        elif value_key == 'CV_ISI_per_energy': 
            value_key_text = '$CV_{ISI}$/$E_{tot}$ (1/($10^{9}$ATP/s)'
        elif value_key == 'MI_tuning_curve': 
            value_key_text = '$MI_{tc}$ (bits)'
        elif value_key == 'MI_tuning_curve_per_energy': 
            value_key_text = '$MI_{tc}$/$E_{tot}$ (bits/($10^{9}$ATP/s)'
        elif value_key == 'MICE_tuning_curve': 
            value_key_text = '$CE_{MI,tc}$ (bits/Hz)'
        elif value_key == 'MICE_tuning_curve_per_energy': 
            value_key_text = '$CE_{MI,tc}$/$E_{tot}$ (s$^{2}$bits/($10^{9}$ATP))'
        elif value_key == 'MI_post': 
            value_key_text = '$MI_{post}$ (bits)'
        elif value_key == 'MI_post_per_energy': 
            value_key_text = '$MI_{post}$/$E_{tot}$ (bits/($10^{9}$ATP/s))'
        elif value_key == 'MICE_post': 
            value_key_text = '$CE_{MI,post}$ (bits/Hz)' # originally '$MI$ CE (bits/Hz)'
        elif value_key == 'MICE_post_per_energy': 
            value_key_text = '$CE_{MI,post}$/$E_{tot}$ (s$^{2}$bits/($10^{9}$ATP))' 
        elif value_key == 'MI': 
            value_key_text = '$MI$ (bits)'
        elif value_key == 'MI_per_energy': 
            value_key_text = '$MI$/$E_{tot}$ (bits/($10^{9}$ATP/s))'
        elif value_key == 'TE': 
            value_key_text = '$TE$ (bits)'
        elif value_key == 'TE_per_energy': 
            value_key_text = '$TE$/$E_{tot}$ (bits/($10^{9}$ATP/s))'
        elif value_key == 'MICE': 
            value_key_text = '$CE_{MI}$ (bits/Hz)' # originally '$MI$ CE (bits/Hz)'
        elif value_key == 'MICE_per_energy': 
            value_key_text = '$CE_{MI}$/$E_{tot}$ (s$^{2}$bits/($10^{9}$ATP))' # originally '$MI$ CE/$E_{tot}$ (s$^{2}$bits/($10^{9}$ATP))'
        elif value_key == 'TECE': 
            value_key_text = '$CE_{TE}$ (bits/Hz)'
        elif value_key == 'TECE_per_energy': 
            value_key_text = '$CE_{TE}$/$E_{tot}$ (s$^{2}$bits/($10^{9}$ATP))'
        elif value_key == 'I_syn_e': 
            value_key_text = '$I_{syn,e}$ (nA)'
        elif value_key == 'R_m': 
            value_key_text = '$R_{m}$ (MOhm)'
        elif value_key == 'E_L': 
            value_key_text = '$V_{rest}$  (mV)'
        elif value_key == 'w_scale': 
            value_key_text = '$W_{e,syn}'
        else: 
            value_key_text = f'{value_key}'
        
        
    return value_key_text

def fit_func_legend_text_plot(fit_func):
    # get string for legend for provided fit_func

    # input
    # fit_func is the function for fitting 

    # output
    # fit_label_template is the corresponding label for the fitting function
    
    if fit_func == linear_func: 
        fit_label_template = 'Fit: $m={:.2e}$, $c={:.2e}$\nEquation: $y = m \\cdot x + c$'
    elif fit_func == sqrt_func: 
        fit_label_template = 'Fit: $m={:.2e}$, $c={:.2e}$\nEquation: $y = m \\cdot \\sqrt{{x}} + c$'
    elif fit_func == log_func: 
        fit_label_template = 'Fit: $a={:.2f}$, $b={:.2f}$, $c={:.2f}$\nEquation: $y = a\log(b \cdot x + 1) + c$'
    elif fit_func == rate_information_func: 
        fit_label_template = 'Fit: $I_{{max}}={:.2f}$, $a={:.2f}$, $n={:.2f}$\nEquation: $y = I_{{max}} (1 - e^{{(-a r^n)}})$'
    elif fit_func == linear_rate_information_func: 
        fit_label_template = 'Fit: $I_{{max}}={:.2f}$, $a={:.2f}$, $n={:.2f}$,\n$m={:.2f}$, $c={:.2f}$\nEquation: $y = I_{{max}} (1 - e^{{(-a x^n)}}) / (m \\cdot x + c)$'
    elif fit_func == sqrt_rate_information_func: 
        fit_label_template = 'Fit: $I_{{max}}={:.2f}$, $a={:.2f}$, $n={:.2f}$,\n$m={:.2f}$, $c={:.2f}$\nEquation: $y = I_{{max}} (1 - e^{{(-a x^n)}}) / (m \\cdot \\sqrt{{x}} + c)$'
    elif fit_func == linear_log_func: 
        fit_label_template = 'Fit: $a={:.2f}$, $b={:.2f}$, $c={:.2f}$, $d={:.2f}$\n Equation: $y = \log(c \cdot x + d) / (a \cdot x + b) $'
    elif fit_func == piecewise_linear_exponential_func: 
        fit_label_template = 'Fit: $k={:.2f}$, $r_{{max}}={:.2f}$, $I_{{max}}={:.2f}$\nEquations:\nfor $x < x_{{max}}$: $I_{{max}} \\cdot x / x_{{max}}$\nfor $x \\geq x_{{max}}$: $I_{{max}} \\cdot e^{{-(x - x_{{max}})/k}}$'
    else: 
        fit_label_template = ' '

    return fit_label_template



def plot_correlation(x, y, x_label, y_label, z=None, z_label=None, fit_func=None, initial_guess=None, params_fit=None, highlight_points=None, fit_func_highlight=None, initial_guess_highlight=None, params_fit_highlight=None, results_grid_point_to_exp_data=None, inverted_x=None, log_log=False, colors=['black', 'red', 'yellow'], show_colorbar=True, plot_mode='correlation', legend_mode=True, figsize=(8, 5), ax=None, savename=None): 
    # plot correlation between two variables with optional fit and color coding

    # input
    # x is an array of x-values to be plotted
    # y is an array of y-values to be plotted
    # x_label is a string to be used as x-axis label
    # y_label is a string to be used as y-axis label
    # z is an array for color coding of scatter points
    # z_label is a string for the z colorbar label
    
    # fit_func is a function to fit the data (e.g. linear_func)
    # a) initial_guess is a list of initial parameters for curve fitting
    # b) params_fit is a list of fixed parameters for curve fitting which skips fitting

    # highlight_points is a tuple of (x_highlight, y_highlight) to be highlighted in yellow
    # fit_func_highlight is a function to fit the highlighted points
    # a) initial_guess_highlight is a initial guess for the highlighted fit
    # b) params_fit_highlight is a fixed parameters for highlighted point fitting
    
    # results_grid_point_to_exp_data is a tuple of arrays of experimental data
    
    # inverted_x is a boolean to invert the x-axis if desired
    # log_log s a boolean to enable loglog scaling if desired
    # colors sets the colors of the CTR, FR & highlight points & fit
    # show_colorbar decides whether to display colorbar or not
    # plot_mode is the mode of x & y labels
    # legend_mode decides if legend should be displayed or not
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # output
    # popt is a list of optimal fit parameters from curve_fit if a fit was performed; otherwise None
    
    color_CTR, color_FR, color_highlight = colors[0], colors[1], colors[2]
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)#
    else:
        fig = ax.get_figure()
    
    # create scatter plot of x & y data with optional color code z
    #scatter = ax.scatter(x, y, s=50, alpha=0.3, color='dimgray' if z is None else None, c=z if z is not None else None, cmap='viridis' if z is not None else None)
    ax.scatter(x, y, s=50, alpha=0.3, color='dimgray' if z is None else None, c=z if z is not None else None, cmap='viridis' if z is not None else None)

    # get correct axis labels
    x_label_text = value_key_text_plot(x_label, plot_mode=plot_mode)
    ax.set_xlabel(x_label_text)
    y_label_text = value_key_text_plot(y_label, plot_mode=plot_mode)
    ax.set_ylabel(y_label_text)
    """if z_label is not None: 
        z_label_text = value_key_text_plot(z_label, plot_mode=plot_mode)
        fig.colorbar(scatter, ax=ax, label=z_label_text) # add colorbar if z is provided"""
    if z is not None and show_colorbar is True:
        if z_label is not None:
            z_label_text = value_key_text_plot(z_label, plot_mode=plot_mode)
        else:
            z_label_text = ""
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=np.min(z), vmax=np.max(z)), cmap='viridis'), ax=ax, label=z_label_text).ax.yaxis.set_major_locator(MaxNLocator(nbins=3)) # make colorbar in full hue & not transparent, constraint 3 ticks
        #fig.colorbar(scatter, ax=ax, label=z_label_text)

    # plot fitting function if provided
    
    # initialize popt
    popt = None

    # plot fitted curve if a fit function is provided either a) perform curve fitting if initial_guess is provided or b) just plot function if params_fit are provded
    if fit_func:
        if initial_guess:
            popt, pcov = curve_fit(fit_func, x, y, p0=initial_guess)
        if params_fit:
            popt = params_fit
        
        x_fit = np.linspace(min(x), max(x), 1000)
        y_fit = fit_func(x_fit, *popt)
        r2 = r2_score(y, fit_func(x, *popt))
        fit_label_template = fit_func_legend_text_plot(fit_func)
        fit_label = fit_label_template.format(*popt) if fit_label_template else 'Fit'
        ax.plot(x_fit, y_fit, label=fit_label, color=plt.cm.viridis(1.0))#'lime')
        
        # add title with correlation value
        if fit_func != None: 
            ax.set_title(f'Correlation: $R^2$-value: {r2:.2f}')
        if fit_func == linear_func:
            # calculate the Pearson correlation coefficient
            correlation, p_value = pearsonr(x, y)
            ax.set_title(f'Correlation\nPearson coef: {correlation:.2f}, p-value: {p_value:.2f}, $R^2$-value: {r2:.2f}')
    
    # plot scatter plot for highlight points if provded
    if highlight_points is not None:
        x_highlight, y_highlight = highlight_points # highlight_points is a tuple of arrays to be highlighted in lime
        ax.scatter(x_highlight, y_highlight, s=50, color=color_highlight, label='highlighted points')
        
        if fit_func_highlight:
            if initial_guess_highlight:
                popt, pcov = curve_fit(fit_func_highlight, x_highlight, y_highlight, p0=initial_guess_highlight)
            if params_fit_highlight:
                popt = params_fit_highlight

            x_fit = np.linspace(0, max(x_highlight), 1000)
            y_fit = fit_func_highlight(x_fit, *popt)
            r2 = r2_score(y_highlight, fit_func_highlight(x_highlight, *popt))
            fit_label_template_highlight = fit_func_legend_text_plot(fit_func_highlight)
            fit_label_highlight = fit_label_template_highlight.format(*popt) if fit_label_template_highlight else 'Fit'
            ax.plot(x_fit, y_fit, label=fit_label_highlight, color=color_highlight)

    # plot closest grid points to experimental data if provided
    
    if results_grid_point_to_exp_data is not None:
        # prepare data
        x_closest_points_CTR = [grid_point["grid_values"].get(f"{x_label}", None) for grid_point in results_grid_point_to_exp_data[0].values()]
        y_closest_points_CTR = [grid_point["grid_values"].get(f"{y_label}", None) for grid_point in results_grid_point_to_exp_data[0].values()]
        x_closest_points_FR = [grid_point["grid_values"].get(f"{x_label}", None) for grid_point in results_grid_point_to_exp_data[1].values()]
        y_closest_points_FR = [grid_point["grid_values"].get(f"{y_label}", None) for grid_point in results_grid_point_to_exp_data[1].values()]
        
        # plot CTR & FR values with all 0.0 values transparent
        
        scatter_size = 25
        
        # determine alpha values for CTR points
        alpha_CTR = [0.2 if x == 0.0 or y == 0.0 else 1.0 for x, y in zip(x_closest_points_CTR, y_closest_points_CTR)]
        # plot values with respective alpha values 
        ax.scatter(x_closest_points_CTR, y_closest_points_CTR, s=scatter_size, color=color_CTR, label='CTR', alpha=alpha_CTR)
        #for x_val, y_val, a in zip(x_closest_points_CTR, y_closest_points_CTR, alpha_CTR): ax.scatter(x_val, y_val, s=50, color=color_FR, alpha=a)
            
        # determine alpha values for FR points
        alpha_FR = [0.2 if x == 0.0 or y == 0.0 else 1.0 for x, y in zip(x_closest_points_FR, y_closest_points_FR)]
        # plot values with respective alpha values
        ax.scatter(x_closest_points_FR, y_closest_points_FR, s=scatter_size, color=color_FR, label='FR', alpha=alpha_FR)
        #for x_val, y_val, a in zip(x_closest_points_FR, y_closest_points_FR, alpha_FR): ax.scatter(x_val, y_val, s=50, color=color_FR, alpha=a)
    
    #plt.ylim(-0.05,0.69)
    if log_log == True: 
        ax.set_xscale('log')
        ax.set_yscale('log') 
    
    if legend_mode is True: 
        # add legend     
        # get current legend handles and labels
        auto_handles, auto_labels = ax.get_legend_handles_labels()
        if auto_handles != []:
            ax.legend(handles=auto_handles, labels=auto_labels, frameon=False)

    # invert x-axis if desired
    if inverted_x is not None:
        #plt.gca().invert_xaxis()
        ax.invert_xaxis()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
        
    # return the optimal parameters if fitting performed
    return popt
            

############################ 3D plotting functions ############################

def find_best_fitting_tuning_curve(results, R_m, E_L, target_tuning_curve):
    # find best fitting tuning curve for closest R_m & E_L values
    
    # input
    # results is a dictionary of the results
    # R_m is the desired membrane resistance in MOhm
    # E_L is the desired leak potential/resting potential in mV
    # target_tuning_curve is a list of the target tuning curve

    # output 
    # closest_key closest key to target tuning curve and R_m & E_L pair 
    # closest_curve is the tuning curve of the closest key
    # closest_coords are the R_m, E_L, w_scale coordinates of the closest match

    # extract constraint (R_m, E_L) pairs from the keys
    available_pairs = []
    key_map = {}
    for key in results.keys():
        try:
            rm_str, el_str, _ = key.split("_")[1], key.split("_")[3], key.split("_")[5]
            rm_val = float(rm_str)
            el_val = float(el_str)
            available_pairs.append((rm_val, el_val))
            key_map[(rm_val, el_val)] = key
        except ValueError:
            pass

    # convert to numpy array for distance computation
    provided_pair = np.array([R_m, E_L])
    available_pairs = np.array(available_pairs)
    distances = np.linalg.norm(available_pairs - provided_pair, axis=1)
    closest_pair = available_pairs[np.argmin(distances)]
    closest_R_m, closest_E_L = closest_pair

    # filter results based on the closest R_m and E_L pair
    closest_key_prefix = f"Rm_{closest_R_m}_EL_{closest_E_L}"
    filtered_results = {k: v for k, v in results.items() if k.startswith(closest_key_prefix)}

    # initialize variables to track the closest match
    min_distance = float('inf')
    closest_key = None
    closest_curve = None

    # compare each tuning curve to the target
    for key, data in filtered_results.items():
        tuning_curve = np.array(data['tuning_curve'])
        distance = np.linalg.norm(tuning_curve - target_tuning_curve) # compute Euclidean distance
        # update if this curve is closer
        if distance < min_distance:
            min_distance = distance
            closest_key = key
            closest_curve = tuning_curve
    
    closest_coords = (float(closest_key.split("_")[1]), float(closest_key.split("_")[3]), float(closest_key.split("_")[5]))
    
    return closest_key, closest_curve, closest_coords

def predict_scaled_w_scale(exp_data, target):
    # fit a linear regression model to experimental data to return w_scale

    #input 
    # exp_data is a tuple of experimental data
    # target is a tuple of desired R_m & E_L & w_scale values

    # output
    # w_scale is the predicted w_scale for the given target
    
    # unpack data
    R_m = exp_data[0]
    E_L = exp_data[1]
    w_scale_0 = exp_data[2]

    # create feature matrix & fit model
    x = np.column_stack((R_m, E_L))
    y = w_scale_0
    model = LinearRegression().fit(x, y)

    # predict unscaled w_scale for the target
    R_m_target, E_L_target, w_scale_0_target = target
    w_scale_pred = model.predict([[R_m_target, E_L_target]])

    # calcualte w_scale by ratio of w_scale_0_target to w_scale_pred
    w_scale_factor = w_scale_0_target / w_scale_pred[0]

    return w_scale_factor

def find_closest_grid_points_to_exp_data(results, exp_data):
    # find the closest grid point for each experimental data point

    # input
    # results is a dictionary of grid run results
    # exp_data is a tuple of experimental data arrays

    # output
    # results_grid_point_to_exp_data is a dictionary where each experimental data point maps to its closest grid point
    
    # extract grid points from keys
    grid_points = []
    for key in results.keys():
        parts = key.split("_")
        R_m = float(parts[1])
        E_L = float(parts[3])
        w_scale = float(parts[5])
        grid_points.append((R_m, E_L, w_scale))
    grid_points = np.array(grid_points) # convert list to array
    
    # initialize dictionary to store matched grid points to exp data points    
    results_grid_point_to_exp_data = {}
    
    R_m_exp = exp_data[0]
    E_L_exp = exp_data[1]
    w_scale_exp = exp_data[2]

    # loop over all exp data points
    for i in range(len(R_m_exp)):
        exp_point = np.array([R_m_exp[i], E_L_exp[i], w_scale_exp[i]])
        
        # compute distances to all grid points
        distances = np.linalg.norm(grid_points - exp_point, axis=1)
        
        # find the index of the closest grid point
        closest_idx = np.argmin(distances)
        closest_grid_point = tuple(grid_points[closest_idx])
        
        # get the grid key for the closest point
        closest_key = f"Rm_{closest_grid_point[0]}_EL_{closest_grid_point[1]}_wscale_{closest_grid_point[2]}"

        # store closest grid point and its value in 
        results_grid_point_to_exp_data[tuple(exp_point)] = {
            "closest_grid_point": closest_grid_point,
            "grid_values": results[closest_key]}

    return results_grid_point_to_exp_data


def find_local_maxima(results, all_trajectories, value_key, non_zero_filtering=False):
    # find end points of trajectories corresponding to 'local maxima'
    
    # input
    # results is the dictionary of saved values 
    # all_trajectories is a list of lists of tuples (3 coordinates) representing the trajectories
    # value_key is the quantity to display, needed in case of non-zero filtering
    # non_zero_filtering filters out all non-zero values

    # output
    # filtered_unique_local_maxima is a list of tuples of unique and potentially non-zero filtered trajectory end points
    # filtered_unique_local_maxima_values is a list of the corresponding values to the filtered trajectory end points
    
    local_maxima = []
    for traj in all_trajectories:
        if not np.isnan(traj[-1]).any(): # only if last entry corresponds to a number the value is considered as a local maximum
            local_maxima.append(traj[-1])

    # convert to a set to remove duplicates and then back to a list
    unique_local_maxima = list(set(local_maxima))

    filtered_unique_local_maxima = []
    filtered_unique_local_maxima_values = []

    # get values of unique_local_maxima
    for rm, el, ws in unique_local_maxima:
        key = f'Rm_{rm}_EL_{el}_wscale_{ws}'
        local_value = results[key][value_key]  # get the value associated with this local maximum
        
        # filter local maxima greater than 0
        if non_zero_filtering == True:
            if local_value > 0:  # only consider values greater than 0
                filtered_unique_local_maxima.append((rm, el, ws))
                filtered_unique_local_maxima_values.append(local_value)
        
        # do not filter local maxima greater than 0
        else: 
            filtered_unique_local_maxima = unique_local_maxima
            filtered_unique_local_maxima_values.append(local_value)

    return filtered_unique_local_maxima, filtered_unique_local_maxima_values


def generate_ellipsoid_data(mean, cov, n_points=100, std_factor=2):
    # generates ellipsoid data points using the mean and covariance matrix
    # input
    # mean is an array of the mean data
    # cov is an array of the covariance matrix of the data
    # n_points is the number of points to generate for the ellipsoid surface
    # std_factor is the scaling factor for the standard deviation (default is 2 for 2 std)

    # output
    # x, y, z are arrays of the coordinates of the ellipsoid surface
    
    # generate grid of points in spherical coordinates
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))

    # decompose the covariance matrix into eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(cov)

    # scale the unit sphere by 2 * square root of the eigenvalues (2 standard deviations)
    scaling_factors = std_factor * np.sqrt(eigenvalues)
    x_ellipsoid = scaling_factors[0] * x_sphere
    y_ellipsoid = scaling_factors[1] * y_sphere
    z_ellipsoid = scaling_factors[2] * z_sphere

    # rotate the ellipsoid by the eigenvectors
    ellipsoid = np.array([x_ellipsoid.ravel(), y_ellipsoid.ravel(), z_ellipsoid.ravel()])
    ellipsoid_rotated = np.dot(eigenvectors, ellipsoid).T

    # shift the ellipsoid to the mean
    x_final = ellipsoid_rotated[:, 0] + mean[0]
    y_final = ellipsoid_rotated[:, 1] + mean[1]
    z_final = ellipsoid_rotated[:, 2] + mean[2]

    return x_final.reshape(n_points, n_points), y_final.reshape(n_points, n_points), z_final.reshape(n_points, n_points)

    

def plot_interactive_3D(results, value_key, title, cmin=None, cmax=None, lower_threshold=None, upper_threshold=None, target_plane=None, iso_energy_OSI_surface=None, interpolation=False, trajectory=None, all_trajectories=None, CTR_FR=None, exp_data=None, plot_ellipsoids=False, plot_exp_stems=False, circled_values=False, colorbar_mode=True, legend_mode=False, axes_mode=False, base_font_size=40, eye=dict(x=-1.8, y=1.6, z=0.8), show_mode=True, savename=None):
    # input
    # results is the dictionary of saved values
    # value_key is the quantity to display color coded
    # title is the title of the plot 
    # cmin is the minimum value of the color code
    # cmax is the maximum value of the color code
    # lower_threshold is the value under which the points should be transparent
    # upper_threshold is the value over which the points should be transparent
    # target_plane is the value for a target energy plane (should be within the upper & lower threshold to be displayed correctly)
    # iso_energy_OSI_surface is the value for a target energy plane with color-coded OSI (should be within the upper & lower threshold to be displayed correctly)
    # interpolation determines if the actual simulated data points should be exchanged by a contionous interpolated plane
    # trajectory is the trajectory connecting a starting point with an end point according to an energy-efficiency scheme
    # all_trajectories are 20 randomly chosen trajectories from all
    # CTR_FR are experimentally measured points to be displayed
    # exp_data are the experimental CTR & FR data points
    # plot_ellipsoids determines if the ellipsoid should be plotted
    # plot_exp_stems plots vertical lines on experimental data to make them better visible 
    # circled_values colors exp_data with color of nearest grid dot
    # colorbar_mode defines if the color bar should be plotted or not
    # legend_mode defines if the legend (CTR & FR) should be plotted or not
    # axes_mode defines if the major axes are plotted with thick lines
    # base_font_size is the base font size
    # eye is the dictionary of the initial camera position
    
    # show_mode determines if the fig gets displayed or not
    # savename is the name to save the grid as html file

    # output
    # fig is the 3D plot
    
    # set label & ticks fontsizes
    
    scaling_factor_colorbar_label = 0.8
    scaling_factor_colorbar_numbers = 0.75 
    scaling_factor_ticks_numbers = 0.5
    scaling_factor_ticks_labels = 0.9
    
    x_label_R_m = 'R<sub>m</sub> (MΩ)'
    y_label_V_rest = 'V<sub>rest</sub> (mV)'
    z_label_w_syn_e = '⟨w<sub>syn,e</sub>⟩ (nS)'
    
    if base_font_size > 49: 
        scaling_factor_colorbar_label = 0.9
        scaling_factor_colorbar_numbers = 1.0
        scaling_factor_ticks_numbers = 0.45
        scaling_factor_ticks_labels = 0.9
        
        x_label_R_m = '  R<sub>m</sub><br>(MΩ)'
        y_label_V_rest = 'V<sub>rest</sub><br>(mV)'
        z_label_w_syn_e = '⟨w<sub>syn,e</sub>⟩<br>  (nS)'
          
    # initialize data
    R_m_values = []
    E_L_values = []
    w_scale_values = []
    r_post_values = []
    E_tot_values = []
    values = []

    for key in results:
        params = key.split('_')
        R_m = float(params[1]) # get R_m value from key
        E_L = float(params[3]) # get E_L value from key
        w_scale = float(params[5]) # get w_scale value from key
        
        R_m_values.append(R_m)
        E_L_values.append(E_L)
        w_scale_values.append(w_scale)
        r_post_value = results[key]['r_post'] # access r_post value from dictionary
        r_post_values.append(r_post_value if r_post_value is not None else 0.0)  # replace None with 0
        E_tot_value = results[key]['E_tot'] # access E_tot value from dictionary
        E_tot_values.append(E_tot_value if E_tot_value is not None else 0.0)  # replace None with 0
        value = results[key][value_key] # access desired value from dictionary
        values.append(value if value is not None else 0.0) # replace None with 0
        
    # calculate default cmin and cmax if not provided
    if cmin is None:
        cmin = min([v for v in values if not np.isnan(v)])
    if cmax is None:
        cmax = max([v for v in values if not np.isnan(v)])

    # create separate lists for zero, below (non-zero) threshold, within thresholds, and above threshold values
    zero_indices = [i for i, v in enumerate(values) if v == 0 or v is None]
    below_threshold_indices = [i for i, v in enumerate(values) if v is not None and v != 0 and lower_threshold is not None and v < lower_threshold]
    within_threshold_indices = [i for i, v in enumerate(values) if v is not None and v != 0 and (lower_threshold is None or v >= lower_threshold) and (upper_threshold is None or v <= upper_threshold)]
    above_threshold_indices = [i for i, v in enumerate(values) if v is not None and v != 0 and upper_threshold is not None and v > upper_threshold]

    zero_R_m_values = [R_m_values[i] for i in zero_indices]
    zero_E_L_values = [E_L_values[i] for i in zero_indices]
    #zero_w_scale_values = [w_scale_values[i] for i in zero_indices] # raw w_scale
    zero_w_scale_values = w_scale_to_w_e_syn([w_scale_values[i] for i in zero_indices]) # translated w_scale to mean excitatory weight 
    zero_r_post_values = [r_post_values[i] for i in zero_indices]
    zero_E_tot_values = [E_tot_values[i] for i in zero_indices]
    
    below_threshold_R_m_values = [R_m_values[i] for i in below_threshold_indices]
    below_threshold_E_L_values = [E_L_values[i] for i in below_threshold_indices]
    #below_threshold_w_scale_values = [w_scale_values[i] for i in below_threshold_indices] # raw w_scale
    below_threshold_w_scale_values = w_scale_to_w_e_syn([w_scale_values[i] for i in below_threshold_indices]) # translated w_scale to mean excitatory weight
    below_threshold_r_post_values = [r_post_values[i] for i in below_threshold_indices]
    below_threshold_E_tot_values = [E_tot_values[i] for i in below_threshold_indices]
    below_threshold_values = [values[i] for i in below_threshold_indices]

    within_threshold_R_m_values = [R_m_values[i] for i in within_threshold_indices]
    within_threshold_E_L_values = [E_L_values[i] for i in within_threshold_indices]
    #within_threshold_w_scale_values = [w_scale_values[i] for i in within_threshold_indices] # raw w_scale
    within_threshold_w_scale_values = w_scale_to_w_e_syn([w_scale_values[i] for i in within_threshold_indices]) # translated w_scale to mean excitatory weight
    within_threshold_r_post_values = [r_post_values[i] for i in within_threshold_indices]
    within_threshold_E_tot_values = [E_tot_values[i] for i in within_threshold_indices]
    within_threshold_values = [values[i] for i in within_threshold_indices]

    above_threshold_R_m_values = [R_m_values[i] for i in above_threshold_indices]
    above_threshold_E_L_values = [E_L_values[i] for i in above_threshold_indices]
    #above_threshold_w_scale_values = [w_scale_values[i] for i in above_threshold_indices] # raw w_scale
    above_threshold_w_scale_values = w_scale_to_w_e_syn([w_scale_values[i] for i in above_threshold_indices]) # translated w_scale to mean excitatory weight
    above_threshold_r_post_values = [r_post_values[i] for i in above_threshold_indices]
    above_threshold_E_tot_values = [E_tot_values[i] for i in above_threshold_indices]
    above_threshold_values = [values[i] for i in above_threshold_indices]
        
    fig = go.Figure()
    
    value_key_text = value_key_text_plot(value_key, plot_mode='grid')
    
    # find the target plane points (e.g. equi-energy plane) if desired
    if target_plane is not None:
        equi_R_m_values = []
        equi_E_L_values = []
        equi_w_scale_values = []
        equi_values = []
        for r_m, e_l in zip(R_m_values, E_L_values):
            min_diff = float('inf')
            closest_w_scale = None
            for key in results:
                params = key.split('_')
                R_m = float(params[1])
                E_L = float(params[3])
                w_scale = float(params[5])
                if R_m == r_m and E_L == e_l:
                    value = results[key][value_key]
                    if value is not None:
                        diff = abs(value - target_plane)
                        if diff < min_diff:
                            min_diff = diff
                            closest_w_scale = w_scale
                            closest_value = value
            if closest_w_scale is not None:
                equi_R_m_values.append(r_m)
                equi_E_L_values.append(e_l)
                #equi_w_scale_values.append(closest_w_scale) # raw w_scale
                equi_w_scale_values.append(w_scale_to_w_e_syn([closest_w_scale])[0]) # translated w_scale to mean excitatory weight
                equi_values.append(closest_value)
        
        # print lower and upper values based on the target plane values
        equi_min = min(equi_values)
        equi_max = max(equi_values)
        
        print(f'Lower value: {equi_min}')
        print(f'Upper value: {equi_max}')
    
        # plot equi-value plane
        fig.add_trace(go.Scatter3d(
            x=equi_R_m_values,
            y=equi_E_L_values,
            z=equi_w_scale_values,
            mode='markers',
            marker=dict(
                size=5,
                color=equi_values,
                colorscale='Viridis',
                showscale=bool(colorbar_mode),
                **({"colorbar": (colorbar_mode if isinstance(colorbar_mode, dict) else {"title": {"text": value_key_text, "font": {"size": base_font_size*scaling_factor_colorbar_label}}, "tickmode":"auto", "nticks":3, "tickfont": {"size": base_font_size*scaling_factor_colorbar_numbers}, "len": 0.5, "thickness": 16, "x": 0.95, "y": 0.425, "yanchor": "middle"})} if colorbar_mode else {}), # "tickmode": "array", "tickvals": np.linspace(cmin, cmax, 3).tolist(), "ticktext": [f"{t:.2f}" for t in np.linspace(cmin, cmax, 3)]
                cmin=cmin,
                cmax=cmax,
                cauto=False,
                opacity=1.0),
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>'))
    
    # find the iso-energy surface points & plot their OSI values if desired
    if iso_energy_OSI_surface is not None:
        iso_R_m_values = []
        iso_E_L_values = []
        iso_w_scale_values = []
        iso_OSI_values = []

        for r_m, e_l in zip(R_m_values, E_L_values):
            min_diff = float('inf')
            closest_w_scale = None
            closest_OSI = None
            for key in results:
                params = key.split('_')
                R_m_k = float(params[1])
                E_L_k = float(params[3])
                w_scale_k = float(params[5])
                if R_m_k == r_m and E_L_k == e_l:
                    E_tot_k = results[key].get('E_tot', None)
                    if E_tot_k is not None:
                        diff = abs(E_tot_k - iso_energy_OSI_surface)
                        if diff < min_diff:
                            min_diff = diff
                            closest_w_scale = w_scale_k
                            closest_OSI = results[key].get('OSI', 0)  # use 0 if OSI is None
            if closest_w_scale is not None:
                iso_R_m_values.append(r_m)
                iso_E_L_values.append(e_l)
                #iso_w_scale_values.append(closest_w_scale) # raw w_scale
                iso_w_scale_values.append(w_scale_to_w_e_syn([closest_w_scale])[0]) # translated w_scale to mean excitatory weight
                iso_OSI_values.append(closest_OSI if closest_OSI is not None else 0.0)
        
        """
        # plot the surface as of grid dots
        fig.add_trace(go.Scatter3d(
            x=iso_R_m_values,
            y=iso_E_L_values,
            z=iso_w_scale_values,
            mode='markers',
            marker=dict(
                size=5,
                color=iso_OSI_values,
                colorscale='Viridis',
                colorbar=dict(title=dict(text='OSI', font=dict(size=base_font_size)), tickfont=dict(size=base_font_size)),
                cmin=cmin,
                cmax=cmax,
                opacity=1.0),
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>OSI: %{marker.color:.2f}<extra></extra>',
            name='Iso-E<sub>tot</sub> OSI plane'))
        """
        
        # plot the surface as of interpolated grid dots (looks almost like surface)
        
        # create meshgrid in R_m-E_L space
        R_m_grid = np.linspace(min(iso_R_m_values), max(iso_R_m_values), 300)
        E_L_grid = np.linspace(min(iso_E_L_values), max(iso_E_L_values), 300)
        R_m_mesh, E_L_mesh = np.meshgrid(R_m_grid, E_L_grid)
        
        # interpolate w_scale and OSI values across the R_m-E_L grid
        grid_w_scale = griddata(
            points=(iso_R_m_values, iso_E_L_values),
            values=iso_w_scale_values,
            xi=(R_m_mesh, E_L_mesh),
            method='linear')
        
        grid_OSI = griddata(
            points=(iso_R_m_values, iso_E_L_values),
            values=iso_OSI_values,
            xi=(R_m_mesh, E_L_mesh),
            method='linear')
        
        # mask out invalid (NaN) regions
        mask = ~np.isnan(grid_w_scale) & ~np.isnan(grid_OSI)
        R_m_plot = R_m_mesh[mask]
        E_L_plot = E_L_mesh[mask]
        w_scale_plot = grid_w_scale[mask]
        OSI_plot = grid_OSI[mask]
        
        # plot interpolated surface as colored 3D scatter
        fig.add_trace(go.Scatter3d(
            x=R_m_plot,
            y=E_L_plot,
            z=w_scale_plot,
            mode='markers',
            marker=dict(
                size=3,
                color=OSI_plot,
                colorscale='Viridis',
                showscale=bool(colorbar_mode),
                **({"colorbar": (colorbar_mode if isinstance(colorbar_mode, dict) else {"title": {"text": value_key_text, "font": {"size": base_font_size*scaling_factor_colorbar_label}}, "tickmode":"auto", "nticks":3, "tickfont": {"size": base_font_size*scaling_factor_colorbar_numbers}, "len": 0.5, "thickness": 16, "x": 0.95, "y": 0.425, "yanchor": "middle"})} if colorbar_mode else {}), # "tickmode": "array", "tickvals": np.linspace(cmin, cmax, 3).tolist(), "ticktext": [f"{t:.2f}" for t in np.linspace(cmin, cmax, 3)]
                cmin=cmin,
                cmax=cmax,
                cauto=False,
                opacity=0.1), #0.05
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>OSI: %{marker.color:.2f}<extra></extra>',
            showlegend=False,
            name='Interpolated OSI surface'))

        """
        # plot the surface as of interpolated volume
        R_m_lin = np.linspace(min(iso_R_m_values), max(iso_R_m_values), 30)
        E_L_lin = np.linspace(min(iso_E_L_values), max(iso_E_L_values), 30)
        w_scale_lin = np.linspace(min(iso_w_scale_values), max(iso_w_scale_values), 30)
        
        R_m_grid, E_L_grid, w_scale_grid = np.meshgrid(R_m_lin, E_L_lin, w_scale_lin)
        
        grid_OSI_values = griddata(
            points=(iso_R_m_values, iso_E_L_values, iso_w_scale_values),
            values=iso_OSI_values,
            xi=(R_m_grid, E_L_grid, w_scale_grid),
            method='linear')
        
        fig.add_trace(go.Volume(
            x=R_m_grid.flatten(),
            y=E_L_grid.flatten(),
            z=w_scale_grid.flatten(),
            value=np.nan_to_num(grid_OSI_values, nan=0).flatten(),
            isomin=cmin,  # OSI color range min
            isomax=cmax,  # OSI color range max
            opacity=0.1,
            surface_count=5,
            colorscale='Viridis',
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>OSI: %{value:.2f}',
            colorbar=dict(title=dict(text='OSI', font=dict(size=base_font_size)), tickfont=dict(size=base_font_size*0.75)),
            name='Interpolated OSI Volume'))"""
        
        # include max OSI dot meaned over 8 nearest neighbors # failed
        """
        import scipy.spatial

        # Step 1: Convert iso-surface coordinates into Nx3 array
        iso_points = np.array(list(zip(iso_R_m_values, iso_E_L_values, iso_w_scale_values)))
        iso_OSI_values_array = np.array(iso_OSI_values)
        
        # Step 2: Find the index of the maximum OSI
        max_osi_index = np.argmax(iso_OSI_values_array)
        max_point = iso_points[max_osi_index]
        
        # Step 3: Use a KDTree to find the 4 nearest neighbors (including itself)
        kdtree = scipy.spatial.cKDTree(iso_points)
        _, neighbor_indices = kdtree.query(max_point, k=8)
        
        # Step 4: Compute mean position and mean OSI across neighbors
        mean_position = iso_points[neighbor_indices].mean(axis=0)
        mean_osi = iso_OSI_values_array[neighbor_indices].mean()
        
        # Step 5: Plot the mean point as a large marker
        fig.add_trace(go.Scatter3d(
            x=[mean_position[0]],
            y=[mean_position[1]],
            z=[mean_position[2]],
            mode='markers',
            marker=dict(
                size=12,
                color=[mean_osi],
                colorscale='Viridis',
                cmin=cmin,
                cmax=cmax,
                line=dict(color='black', width=2)
            ),
            name='Max OSI region',
            hovertemplate='R_m: %{x:.2f}<br>E_L: %{y:.2f}<br>w_syn_e: %{z:.2f}<br>Mean OSI (4-NN): %{marker.color:.2f}<extra></extra>'
        ))"""


    # plot interpolated plane instead of discrete points if enabled
    if interpolation:
        # define grid for interpolation
        R_m_grid = np.linspace(min(R_m_values), max(R_m_values), 30)
        E_L_grid = np.linspace(min(E_L_values), max(E_L_values), 30)
        #w_scale_grid = np.linspace(min(w_scale_values), max(w_scale_values), 30) # raw w_scale
        w_scale_nS = w_scale_to_w_e_syn(w_scale_values) # translated w_scale to mean excitatory weight
        w_scale_grid = np.linspace(min(w_scale_nS), max(w_scale_nS), 30) # translated w_scale to mean excitatory weight
        
        # prepare meshgrid
        grid_R_m, grid_E_L, grid_w_scale = np.meshgrid(R_m_grid, E_L_grid, w_scale_grid)
        
        # interpolate using griddata
        interpolated_values = griddata((within_threshold_R_m_values, within_threshold_E_L_values, within_threshold_w_scale_values), within_threshold_values, (grid_R_m, grid_E_L, grid_w_scale), method='linear')
        
        # plot interpolated volume as isosurface
        fig.add_trace(go.Volume(
            x=grid_R_m.flatten(),
            y=grid_E_L.flatten(),
            z=grid_w_scale.flatten(),
            value=interpolated_values.flatten(),
            isomin=lower_threshold,
            isomax=upper_threshold,
            opacity=0.1,
            surface_count=5,
            colorscale='Viridis',
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>',
            showlegend=False, 
            showscale=bool(colorbar_mode),
            **({"colorbar": (colorbar_mode if isinstance(colorbar_mode, dict) else {"title": {"text": value_key_text, "font": {"size": base_font_size*scaling_factor_colorbar_label}}, "tickmode":"auto", "nticks":3, "tickfont": {"size": base_font_size*scaling_factor_colorbar_numbers}, "len": 0.5, "thickness": 16, "x": 0.95, "y": 0.425, "yanchor": "middle"})} if colorbar_mode else {})# "tickmode": "array", "tickvals": np.linspace(cmin, cmax, 3).tolist(), "ticktext": [f"{t:.2f}" for t in np.linspace(cmin, cmax, 3)]
            ))
            #colorbar=dict(title=dict(text=value_key_text, font=dict(size=base_font_size)), tickfont=dict(size=base_font_size*0.75))))
        
    # plot and calculate grid if not target_plane or iso_energy_OSI_surface or interpolation desired
    #elif target_plane is None or interpolation is False:
    elif target_plane is None and iso_energy_OSI_surface is None and not interpolation:

        
        # plot within threshold values
        fig.add_trace(go.Scatter3d(
            x=within_threshold_R_m_values,
            y=within_threshold_E_L_values,
            z=within_threshold_w_scale_values,
            customdata=np.column_stack((within_threshold_r_post_values, within_threshold_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color=within_threshold_values,
                colorscale='Viridis',
                showscale=bool(colorbar_mode),
                **({"colorbar": (colorbar_mode if isinstance(colorbar_mode, dict) else {"title": {"text": value_key_text, "font": {"size": base_font_size*scaling_factor_colorbar_label}}, "tickmode":"auto", "nticks":3, "tickfont": {"size": base_font_size*scaling_factor_colorbar_numbers}, "len": 0.5, "thickness": 16, "x": 0.95, "y": 0.425, "yanchor": "middle"})} if colorbar_mode else {}), # "tickmode": "array", "tickvals": np.linspace(cmin, cmax, 3).tolist(), "ticktext": [f"{t:.2f}" for t in np.linspace(cmin, cmax, 3)]
                #colorbar=dict(title=dict(text=value_key_text, font=dict(size=base_font_size)), tickfont=dict(size=base_font_size*0.75)), 
                cmin=cmin,
                cmax=cmax,
                cauto=False,
                opacity=1.0),
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            showlegend=False, 
            name='Within thresholds'))

        # plot zero values as empty circles
        fig.add_trace(go.Scatter3d(
            x=zero_R_m_values,
            y=zero_E_L_values,
            z=zero_w_scale_values,
            customdata=np.column_stack((zero_r_post_values, zero_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(0,0,0,0)',  # make them transparent
                line=dict(color='black', width=0.5)),  # black outline
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': 0',
            showlegend=False,
            name='Zero values'))

        # plot below threshold values in color but with low opacity
        fig.add_trace(go.Scatter3d(
            x=below_threshold_R_m_values,
            y=below_threshold_E_L_values,
            z=below_threshold_w_scale_values,
            customdata=np.column_stack((below_threshold_r_post_values, below_threshold_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color=below_threshold_values,
                colorscale='Viridis',
                showscale=bool(colorbar_mode),
                **({"colorbar": (colorbar_mode if isinstance(colorbar_mode, dict) else {"title": {"text": value_key_text, "font": {"size": base_font_size*scaling_factor_colorbar_label}}, "tickmode":"auto", "nticks":3, "tickfont": {"size": base_font_size*scaling_factor_colorbar_numbers}, "len": 0.5, "thickness": 16, "x": 0.95, "y": 0.425, "yanchor": "middle"})} if colorbar_mode else {}), # "tickmode": "array", "tickvals": np.linspace(cmin, cmax, 3).tolist(), "ticktext": [f"{t:.2f}" for t in np.linspace(cmin, cmax, 3)]
                cmin=cmin,
                cmax=cmax,
                cauto=False,
                opacity=0.1),
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            showlegend=False, 
            name='Below threshold values'))

        # plot above threshold values in color but with low opacity
        fig.add_trace(go.Scatter3d(
            x=above_threshold_R_m_values,
            y=above_threshold_E_L_values,
            z=above_threshold_w_scale_values,
            customdata=np.column_stack((above_threshold_r_post_values, above_threshold_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color=above_threshold_values,
                colorscale='Viridis',
                showscale=bool(colorbar_mode),
                **({"colorbar": (colorbar_mode if isinstance(colorbar_mode, dict) else {"title": {"text": value_key_text, "font": {"size": base_font_size*scaling_factor_colorbar_label}}, "tickmode":"auto", "nticks":3, "tickfont": {"size": base_font_size*scaling_factor_colorbar_numbers}, "len": 0.5, "thickness": 16, "x": 0.95, "y": 0.425, "yanchor": "middle"})} if colorbar_mode else {}), # "tickmode": "array", "tickvals": np.linspace(cmin, cmax, 3).tolist(), "ticktext": [f"{t:.2f}" for t in np.linspace(cmin, cmax, 3)]
                cmin=cmin,
                cmax=cmax,
                cauto=False,
                opacity=0.1),
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            showlegend=False,
            name='Above threshold values'))

    # plot a single trajectory as a red line
    if trajectory is not None:
        traj_R_m, traj_E_L, traj_w_scale = zip(*trajectory)
        traj_w_scale = w_scale_to_w_e_syn(traj_w_scale) # translated w_scale to mean excitatory weight
        fig.add_trace(go.Scatter3d(
            x=traj_R_m,
            y=traj_E_L,
            z=traj_w_scale,
            mode='lines', # with markers: 'lines+markers'
            line=dict(color='red', width=4),
            marker=dict(size=4, color='red'),
            name='single trajectory'))
    
    # if all_trajectories is provided plot 20 random example trajectories 
    if all_trajectories is not None:
        # plot 20 random example trajectories from all_trajectories
        example_trajectories = random.sample(all_trajectories, 20)  # Select 20 random trajectories


        for traj in example_trajectories:
            traj_R_m, traj_E_L, traj_w_scale = zip(*traj)
            traj_w_scale = w_scale_to_w_e_syn(traj_w_scale) # translated w_scale to mean excitatory weight
            traj_values = [results[f'Rm_{rm}_EL_{el}_wscale_{ws}'][value_key] for rm, el, ws in traj]
            fig.add_trace(go.Scatter3d(
                x=traj_R_m,
                y=traj_E_L,
                z=traj_w_scale,
                mode='lines',
                line=dict(
                    color=traj_values,
                    colorscale='Viridis',
                    cmin=cmin,
                    cmax=cmax,
                    cauto=False,
                    width=6),
                opacity=0.95,
                showlegend=False,
                name='Example Trajectories'))

        # find local maxima
        filtered_unique_local_maxima, filtered_unique_local_maxima_values = find_local_maxima(results, all_trajectories, value_key, non_zero_filtering=False)

        # check if there are any valid maxima left after filtering and plot them
        if filtered_unique_local_maxima:
            # unpack the filtered local maxima values for plotting
            local_R_m, local_E_L, local_w_scale = zip(*filtered_unique_local_maxima)
            local_w_scale = w_scale_to_w_e_syn(local_w_scale) # translated w_scale to mean excitatory weight 

            # plot filtered local maxima
            fig.add_trace(go.Scatter3d(
                x=local_R_m,
                y=local_E_L,
                z=local_w_scale,
                mode='markers',
                marker=dict(
                    size=8,
                    color=filtered_unique_local_maxima_values,  # use the filtered values for coloring
                    colorscale='Viridis',
                    cmin=cmin,
                    cmax=cmax,
                    cauto=False,
                    opacity=1.0),
                hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
                showlegend=False,
                name='Local Maxima'))
        
    # plot mean CTR in black and mean FR point in dark red
    if CTR_FR is not None:
        CTR_coords, FR_coords = CTR_FR
        CTR_R_m, CTR_E_L, CTR_w_scale = CTR_coords
        FR_R_m, FR_E_L, FR_w_scale = FR_coords
        CTR_w_scale = w_scale_to_w_e_syn([CTR_w_scale])[0] # translated w_scale to mean excitatory weight
        FR_w_scale = w_scale_to_w_e_syn([FR_w_scale])[0] # translated w_scale to mean excitatory weight
        
        fig.add_trace(go.Scatter3d(
            x=[CTR_R_m],
            y=[CTR_E_L],
            z=[CTR_w_scale],
            mode='markers',
            marker=dict(
                size=8,
                color='black'),
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            showlegend=False,
            name='CTR Point'))
        
        fig.add_trace(go.Scatter3d(
            x=[FR_R_m],
            y=[FR_E_L],
            z=[FR_w_scale],
            mode='markers',
            marker=dict(
                size=8,
                color='darkred'
                ),
            hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            showlegend=False,
            name='FR Point'))

    # plot experimental data if provided
    if exp_data is not None:# and (len(exp_data[0][0]) > 0 or len(exp_data[1][0]) > 0):
        exp_data_CTR = exp_data[0]
        exp_data_FR = exp_data[1]
        R_m_CTR, E_L_CTR, w_scale_CTR = exp_data_CTR[0], exp_data_CTR[1], exp_data_CTR[2]
        R_m_FR, E_L_FR, w_scale_FR = exp_data_FR[0], exp_data_FR[1], exp_data_FR[2]
        w_scale_CTR = w_scale_to_w_e_syn(w_scale_CTR) # translated w_scale to mean excitatory weight
        w_scale_FR = w_scale_to_w_e_syn(w_scale_FR) # translated w_scale to mean excitatory weight
        mean_CTR = np.array([np.mean(R_m_CTR), np.mean(E_L_CTR), np.mean(w_scale_CTR)])
        mean_FR = np.array([np.mean(R_m_FR), np.mean(E_L_FR), np.mean(w_scale_FR)])

        # plot CTR and FR scatter points
        fig.add_trace(go.Scatter3d(x=R_m_CTR, y=E_L_CTR, z=w_scale_CTR, mode='markers', marker=dict(size=5, color='black'), hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>', showlegend=legend_mode, name='CTR data'))
        fig.add_trace(go.Scatter3d(x=R_m_FR, y=E_L_FR, z=w_scale_FR, mode='markers', marker=dict(size=5, color='red'), hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>', showlegend=legend_mode, name='FR data'))

        # plot mean CTR and FR scatter points
        #fig.add_trace(go.Scatter3d(x=[mean_CTR[0]], y=[mean_CTR[1]], z=[mean_CTR[2]], mode='markers', marker=dict(size=10, color='black'), hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>', showlegend=False, name='Mean CTR data'))
        #fig.add_trace(go.Scatter3d(x=[mean_FR[0]], y=[mean_FR[1]], z=[mean_FR[2]], mode='markers', marker=dict(size=10, color='red'), hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>', showlegend=False, name='Mean FR data'))
        
        # plot colored inside
        if iso_energy_OSI_surface is not None and circled_values:
            # map data
            closest_CTR = find_closest_grid_points_to_exp_data(results, (exp_data_CTR[0], exp_data_CTR[1], exp_data_CTR[2]))
            closest_FR  = find_closest_grid_points_to_exp_data(results, (exp_data_FR[0],  exp_data_FR[1],  exp_data_FR[2]))
        
            # overlay smaller filled faces colored by nearest grid point's OSI
            for x, y, z_raw, z_nS in zip(exp_data_CTR[0], exp_data_CTR[1], exp_data_CTR[2], w_scale_CTR):
                info = closest_CTR[(x, y, z_raw)]
                key  = f"Rm_{info['closest_grid_point'][0]}_EL_{info['closest_grid_point'][1]}_wscale_{info['closest_grid_point'][2]}"
                face = results[key].get('OSI', 0.0)
                fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z_nS], mode='markers',
                    marker=dict(size=3, color=[face], colorscale='Viridis', cmin=cmin, cmax=cmax, cauto=False),
                    hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>OSI: %{marker.color:.2f}<extra></extra>', showlegend=False))
        
            for x, y, z_raw, z_nS in zip(exp_data_FR[0], exp_data_FR[1], exp_data_FR[2], w_scale_FR):
                info = closest_FR[(x, y, z_raw)]
                key  = f"Rm_{info['closest_grid_point'][0]}_EL_{info['closest_grid_point'][1]}_wscale_{info['closest_grid_point'][2]}"
                face = results[key].get('OSI', 0.0)
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z_nS], mode='markers',
                    marker=dict(size=3, color=[face], colorscale='Viridis', cmin=cmin, cmax=cmax, cauto=False),
                    hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>OSI: %{marker.color:.2f}<extra></extra>', showlegend=False))
        
        if plot_exp_stems is True:
            
            floor_z = min(w_scale_to_w_e_syn(w_scale_values))  # or 0.0 if preferred
    
            # plot stems for CTR data
            for x, y, z in zip(R_m_CTR, E_L_CTR, w_scale_CTR):
                fig.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[floor_z, z], mode='lines', line=dict(color='black', width=4, dash='dot'), showlegend=False, hoverinfo='skip'))
    
            # plot stems for FR data
            for x, y, z in zip(R_m_FR, E_L_FR, w_scale_FR):
                fig.add_trace(go.Scatter3d(x=[x, x], y=[y, y], z=[floor_z, z], mode='lines', line=dict(color='red', width=4, dash='dot'), showlegend=False, hoverinfo='skip'))
            
    # plot ellipsoid if desired
    if plot_ellipsoids and exp_data is not None:
        exp_data_CTR = exp_data[0]
        exp_data_FR = exp_data[1]
        R_m_CTR, E_L_CTR, w_scale_CTR = exp_data_CTR[0], exp_data_CTR[1], exp_data_CTR[2]
        R_m_FR, E_L_FR, w_scale_FR = exp_data_FR[0], exp_data_FR[1], exp_data_FR[2]
        w_scale_CTR = w_scale_to_w_e_syn(w_scale_CTR) # translated w_scale to mean excitatory weight
        w_scale_FR = w_scale_to_w_e_syn(w_scale_FR) # translated w_scale to mean excitatory weight
        
        # get CTR ellipsoid parameters
        mean_CTR = np.array([np.mean(R_m_CTR), np.mean(E_L_CTR), np.mean(w_scale_CTR)])
        cov_CTR = np.cov(np.vstack((R_m_CTR, E_L_CTR, w_scale_CTR)))
        x_CTR, y_CTR, z_CTR = generate_ellipsoid_data(mean_CTR, cov_CTR, std_factor=2)
        mask_CTR = z_CTR >= 0
        x_CTR, y_CTR, z_CTR = [np.where(mask_CTR, axis, np.nan) for axis in (x_CTR, y_CTR, z_CTR)]
        
        # get FR ellipsoid parameters
        mean_FR = np.array([np.mean(R_m_FR), np.mean(E_L_FR), np.mean(w_scale_FR)])
        cov_FR = np.cov(np.vstack((R_m_FR, E_L_FR, w_scale_FR)))
        x_FR, y_FR, z_FR = generate_ellipsoid_data(mean_FR, cov_FR, std_factor=2)
        mask_FR = z_FR >= 0
        x_FR, y_FR, z_FR = [np.where(mask_FR, axis, np.nan) for axis in (x_FR, y_FR, z_FR)]
             
        # plot CTR and FR ellipsoids
        fig.add_trace(go.Surface(x=x_CTR, y=y_CTR, z=z_CTR, colorscale=[[0, 'rgba(0, 0, 0, 0.2)'], [1, 'rgba(0, 0, 0, 0.2)']], hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>', showscale=False, showlegend=False, name='CTR Ellipsoid'))
        fig.add_trace(go.Surface(x=x_FR, y=y_FR, z=z_FR, colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(255, 0, 0, 0.2)']], hovertemplate='R_m: %{x}<br>E_L: %{y}<br>w_syn_e: %{z}<br>', showscale=False, showlegend=False, name='FR Ellipsoid'))
        
    # color bar settings
    
    # enforce size globally based on colorbar_mode dicts
    if colorbar_mode:
        len_scale = 0.75
        thick_scale = 20
        for tr in fig.data:
            if hasattr(tr, "colorbar") and tr.colorbar is not None:
                tr.colorbar.len = len_scale
                tr.colorbar.thickness = thick_scale
        if "coloraxis" in fig.layout:
            fig.layout.coloraxis.colorbar.len = len_scale
            fig.layout.coloraxis.colorbar.thickness = thick_scale
    

    # update layout for the 3D plot
    
    """
    fig.update_layout(
    scene=dict(
        xaxis=dict(
            title=dict(
                text='R<sub>m</sub> (MΩ)',  # x-axis label
                font=dict(size=base_font_size)), tickfont=dict(size=base_font_size*0.5), tickmode='auto', nticks=3, backgroundcolor='white', gridcolor='lightgrey', showbackground=True), #tickmode='linear', dtick=35
        yaxis=dict(
            title=dict(
                text='V<sub>rest</sub> (mV'),  # y-axis label
                font=dict(size=base_font_size)), tickfont=dict(size=base_font_size*0.5), tickmode='auto', nticks=3, backgroundcolor='white', gridcolor='lightgrey', showbackground=True), #tickmode='linear', dtick=19, 
        zaxis=dict(
            title=dict(
                #text='w<sub>scale</sub>',  # z-axis label # raw w_scale
                text='⟨w<sub>syn,e</sub>⟩ (nS)', # z-axis label 
                font=dict(size=base_font_size)), tickfont=dict(size=base_font_size*0.5), tickmode='auto', nticks=3, backgroundcolor='white', gridcolor='lightgrey', showbackground=True, range=[min(w_scale_to_w_e_syn(w_scale_values)), max(w_scale_to_w_e_syn(w_scale_values))]), #tickmode='linear', dtick=0.4,
                """
        
    # get min/max of axis
    x_min, x_max = int(min(R_m_values)), int(max(R_m_values))
    y_min, y_max = int(min(E_L_values)), int(max(E_L_values))
    z_min, z_max = min(w_scale_to_w_e_syn(w_scale_values)), max(w_scale_to_w_e_syn(w_scale_values))
    
    # position ticks at 10% & 90% of the length
    def edge_ticks(vmin, vmax):
        lower = vmin + 0.1 * (vmax - vmin)
        upper = vmin + 0.9 * (vmax - vmin)
        return [lower, upper]
    
    x_ticks = edge_ticks(x_min, x_max)
    y_ticks = edge_ticks(y_min, y_max)
    z_ticks = edge_ticks(z_min, z_max)

    # thicker axis line style if enabled by axes_mode
    axis_line_width = 8 if axes_mode else 2
    #axis_show_line  = True if axes_mode else False  # explicitly show the axis line when enabled
    
    # explicit axis lines using the tick positions
    if axes_mode:
        x1, x0 = x_ticks
        x1, x0 = x1/1.1, x0/0.9
        y1, y0 = y_ticks
        y1, y0 = y1/0.95, y0/1.05
        z0, z1 = z_ticks
        z0, z1 = z0*0.1, z1/0.99

        # choose one "corner" for the axes to originate from
        # here: (x0, y0, z0), i.e. 10% into each dimension
        fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y0], z=[z0, z0], mode='lines', line=dict(width=axis_line_width, color='black'), hoverinfo='skip', showlegend=False, name='x-axis'))
        fig.add_trace(go.Scatter3d(x=[x0, x0], y=[y0, y1], z=[z0, z0], mode='lines', line=dict(width=axis_line_width, color='black'), hoverinfo='skip', showlegend=False, name='y-axis'))
        fig.add_trace(go.Scatter3d(x=[x0, x0], y=[y0, y0], z=[z0, z1], mode='lines', line=dict(width=axis_line_width, color='black'), hoverinfo='skip', showlegend=False, name='z-axis'))

    
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text=x_label_R_m, font=dict(size=base_font_size*scaling_factor_ticks_labels)), # with <br> for linebreak
                tickvals=x_ticks,
                ticktext=[f"{x_ticks[0]:.0f}", f"{x_ticks[1]:.0f}"],
                tickfont=dict(size=base_font_size*scaling_factor_ticks_numbers),
                showbackground=True, backgroundcolor='white', gridcolor='lightgrey'), #, showline=axis_show_line, linecolor='black', linewidth=axis_line_width
            yaxis=dict(
                title=dict(text=y_label_V_rest, font=dict(size=base_font_size*scaling_factor_ticks_labels)), # with <br> for linebreak
                tickvals=y_ticks,
                ticktext=[f"{y_ticks[0]:.0f}", f"{y_ticks[1]:.0f}"],
                tickfont=dict(size=base_font_size*scaling_factor_ticks_numbers),
                showbackground=True, backgroundcolor='white', gridcolor='lightgrey'), #, showline=axis_show_line, linecolor='black', linewidth=axis_line_width
            zaxis=dict(
                title=dict(text=z_label_w_syn_e, font=dict(size=base_font_size*scaling_factor_ticks_labels)), # with <br> for linebreak
                tickvals=z_ticks,
                ticktext=[f"{z_ticks[0]:.2f}", f"{z_ticks[1]:.2f}"],
                tickfont=dict(size=base_font_size*scaling_factor_ticks_numbers),
                showbackground=True, backgroundcolor='white', gridcolor='lightgrey'), #, showline=axis_show_line, linecolor='black', linewidth=axis_line_width
            aspectmode='cube'),
    
            #title=title,
            width=850,
            height=850,
            scene_camera=dict(eye=eye),
    
            font=dict(
                family='Helvetica', #'CMU Serif',  #  only safely used with base_font_size = 20, LaTeX-like font 
                color='black'),
                legend=dict(
                    yanchor='top',
                    y=0.1,
                    xanchor='left',
                    x=0.05),
            margin=dict(l=0, r=0, t=0, b=0))
            
    
    if show_mode is True: 
        fig.show()

    # save the plot as an HTML file
    if savename is not None:
        pio.write_html(fig, f"../Figures/{savename}.html") # interactive html
        pio.write_image(fig, f"../Figures/{savename}.pdf", scale=2) # saves static pdf (requires kaleido)
        pio.write_image(fig, f"../Figures/{savename}.png", scale=2) # saves static png (requires kaleido)
    
    return fig


# plot 3D matplotlib




def parse_results_grid(results, value_key):
    # parse results dict into parameter arrays and value arrays
    # input
    # results is a dict with keys like 'Rm_{rm}_EL_{el}_wscale_{ws}' and values as dicts containing 'r_post', 'E_tot', and value_key
    # value_key is the dict key for the scalar quantity to extract from each results entry
    # output
    # Rm is an array of Rm values
    # EL is an array of EL values
    # wscale is an array of raw wscale values
    # r_post is an array of r_post values (0.0 if missing)
    # E_tot is an array of E_tot values (0.0 if missing)
    # values is an array of scalar values for value_key (0.0 if missing)
    Rm, EL, wscale, r_post, E_tot, values = [], [], [], [], [], []

    for k, d in results.items():
        parts = k.split('_')
        rm = float(parts[1])
        el = float(parts[3])
        ws = float(parts[5])

        Rm.append(rm)
        EL.append(el)
        wscale.append(ws)

        rp = d.get('r_post', None)
        et = d.get('E_tot', None)
        vv = d.get(value_key, None)

        r_post.append(0.0 if rp is None else float(rp))
        E_tot.append(0.0 if et is None else float(et))
        values.append(0.0 if vv is None else float(vv))

    return np.array(Rm), np.array(EL), np.array(wscale), np.array(r_post), np.array(E_tot), np.array(values)


def edge_ticks(vmin, vmax, frac_low=0.1, frac_high=0.9):
    # compute two tick positions near the lower and upper edges of an interval
    # input
    # vmin is the minimum axis value
    # vmax is the maximum axis value
    # frac_low is the fraction from the lower edge for the first tick
    # frac_high is the fraction from the lower edge for the second tick
    # output
    # ticks is a list of two tick locations
    lower = vmin + frac_low * (vmax - vmin)
    upper = vmin + frac_high * (vmax - vmin)
    return [lower, upper]


def build_index_by_RmEL(results, value_key):
    # build index mapping (Rm, EL) to list of entries for different wscale values
    # input
    # results is a dict with keys like 'Rm_{rm}_EL_{el}_wscale_{ws}'
    # value_key is the key used to store the scalar in the entry field 'value'
    # output
    # idx is a dict mapping (rm, el) to a list of dict entries containing wscale, value, E_tot, OSI, and r_post
    idx = {}
    for k, d in results.items():
        parts = k.split('_')
        rm = float(parts[1])
        el = float(parts[3])
        ws = float(parts[5])

        entry = dict(wscale=ws, value=d.get(value_key, None), E_tot=d.get('E_tot', None), OSI=d.get('OSI', None), r_post=d.get('r_post', None))
        idx.setdefault((rm, el), []).append(entry)
    return idx


def get_entry_closest_wscale(idx_by_RmEL, rm, el, wscale_target, value_key):
    # return closest wscale entry for a given (rm, el) and its scalar value
    # input
    # idx_by_RmEL is the dict produced by build_index_by_RmEL
    # rm is the Rm coordinate
    # el is the EL coordinate
    # wscale_target is the target wscale to match within the (rm, el) column
    # value_key selects which scalar to return, either 'OSI' or the value stored in entry['value']
    # output
    # wscale_closest is the closest wscale in that (rm, el) column
    # v_closest is the scalar value at that entry, 0.0 if missing or non-finite
    entries = idx_by_RmEL.get((rm, el), None)
    if not entries:
        return None, None

    best = None
    best_diff = np.inf
    for e in entries:
        ws = e.get('wscale', None)
        if ws is None or not np.isfinite(ws):
            continue
        diff = abs(ws - wscale_target)
        if diff < best_diff:
            best_diff = diff
            best = e

    if best is None:
        return None, None

    v = best.get('OSI', None) if value_key == 'OSI' else best.get('value', None)
    v_closest = 0.0 if v is None or not np.isfinite(v) else float(v)
    return best.get('wscale', None), v_closest


def add_colored_trajectory_3d(ax, traj, idx_value, norm, cmap, w_scale_to_w_e_syn, linewidth=4, alpha=0.95, show_nodes=True, node_size=18, node_alpha=0.95, node_edgecolor="none", node_linewidth=0.0):
    # plot a 3d trajectory with per-segment color coding and optional node dots
    # input
    # ax is a matplotlib 3d axes
    # traj is a list of (Rm, EL, wscale_raw) points
    # idx_value is the index from build_index_by_RmEL using the plotted value_key
    # norm is a matplotlib Normalize instance for scalar to color mapping
    # cmap is a matplotlib colormap name or instance
    # w_scale_to_w_e_syn maps raw wscale to mean excitatory synaptic weight in nS
    # linewidth is the line width for trajectory segments
    # alpha is the line alpha for trajectory segments
    # show_nodes determines if node dots are plotted
    # node_size is the marker size for node dots
    # node_alpha is the marker alpha for node dots
    # node_edgecolor is the marker edge color for node dots
    # node_linewidth is the marker edge linewidth for node dots
    # output
    # sc is the scatter artist for node dots or None
    # lc is the Line3DCollection artist for colored segments
    if traj is None or len(traj) < 2:
        return None, None

    # convert trajectory points to xyz coordinates
    pts = np.asarray([[rm, el, w_scale_to_w_e_syn([ws])[0]] for (rm, el, ws) in traj])

    # compute scalar value at each node using closest wscale match in the (rm, el) column
    node_vals = []
    for (rm, el, ws) in traj:
        entries = idx_value.get((rm, el), None)
        if not entries:
            node_vals.append(np.nan)
            continue
        best = min(entries, key=lambda e: abs(e["wscale"] - ws))
        v = best.get("value", None)
        node_vals.append(0.0 if (v is None or not np.isfinite(v)) else float(v))
    node_vals = np.asarray(node_vals)

    # plot nodes first so the line drawn afterwards looks smooth
    sc = None
    if show_nodes:
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=node_vals, cmap=cmap, norm=norm, s=node_size, alpha=node_alpha, depthshade=False, edgecolors=node_edgecolor, linewidths=node_linewidth, zorder=2500)

    # plot colored segments on top, colored by the origin node value
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_vals = node_vals[:-1]
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(seg_vals)
    ax.add_collection(lc)

    return sc, lc


def plot_volume_slices(ax, fig, x3, y3, z3, v3, norm, cmap, n_slices=12, alpha=0.08, stride=1, colorbar=True, cbar_label="", cbar_kwargs=None):
    # render a volume-like visualization by plotting multiple semi-transparent xy slices
    # input
    # ax is a matplotlib 3d axes
    # fig is the matplotlib figure
    # x3 is a 3d array of x coordinates
    # y3 is a 3d array of y coordinates
    # z3 is a 3d array of z coordinates
    # v3 is a 3d array of scalar values to color the slices
    # norm is a matplotlib Normalize instance for scalar to color mapping
    # cmap is a matplotlib colormap name or instance
    # n_slices is the number of z slices to render
    # alpha is the alpha of each slice surface
    # stride is a downsampling step for x and y to reduce load
    # colorbar determines whether to add a colorbar
    # cbar_label is the label string for the colorbar
    # cbar_kwargs is a dict of kwargs passed to fig.colorbar
    # output
    # cbar is the created colorbar or None
    if cbar_kwargs is None:
        cbar_kwargs = dict(shrink=0.4, pad=0.15, aspect=20)

    nz = v3.shape[2]
    slice_ids = np.linspace(0, nz - 1, n_slices).astype(int)
    cm = plt.get_cmap(cmap)

    for k in slice_ids:
        X = x3[:, :, k]
        Y = y3[:, :, k]
        Z = z3[:, :, k]
        V = v3[:, :, k]

        mask = np.isfinite(V)
        if mask.sum() < 20:
            continue

        Xp = X[::stride, ::stride]
        Yp = Y[::stride, ::stride]
        Zp = Z[::stride, ::stride]
        Vp = V[::stride, ::stride]

        fc = cm(norm(Vp))
        fc[..., 3] = alpha

        ax.plot_surface(Xp, Yp, Zp, facecolors=fc, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

    if colorbar:
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(v3[np.isfinite(v3)])
        cbar = fig.colorbar(mappable, ax=ax, **cbar_kwargs)
        cbar.set_label(cbar_label)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        return cbar

    return None


def plot_isosurface_from_volume(ax, vol, x_lin, y_lin, z_lin, level, norm, cmap="viridis", alpha=0.25):
    # plot a single isosurface extracted from a 3d scalar field using marching cubes
    # input
    # ax is a matplotlib 3d axes
    # vol is a 3d array scalar field
    # x_lin is the x coordinate linspace matching vol axis 0
    # y_lin is the y coordinate linspace matching vol axis 1
    # z_lin is the z coordinate linspace matching vol axis 2
    # level is the isovalue at which to extract the surface
    # norm is a matplotlib Normalize instance for scalar to color mapping
    # cmap is a matplotlib colormap name or instance
    # alpha is the surface alpha
    # output
    # mesh is a Poly3DCollection or None if level is outside the volume range
    if marching_cubes is None:
        raise RuntimeError("scikit-image required for isosurfaces")

    vmin = np.nanmin(vol)
    vmax = np.nanmax(vol)
    if not (vmin <= level <= vmax):
        return None

    verts_ijk, faces, _, _ = marching_cubes(vol, level=float(level))

    nx, ny, nz = vol.shape
    xi = np.interp(verts_ijk[:, 0], np.arange(nx), x_lin)
    yi = np.interp(verts_ijk[:, 1], np.arange(ny), y_lin)
    zi = np.interp(verts_ijk[:, 2], np.arange(nz), z_lin)
    verts_xyz = np.column_stack([xi, yi, zi])

    cm = plt.get_cmap(cmap)
    facecolor = cm(norm(level))

    mesh = Poly3DCollection(verts_xyz[faces], facecolor=facecolor, edgecolor="none", alpha=alpha)
    ax.add_collection3d(mesh)
    return mesh


def plot_grid_3D_matplotlib(results, value_key, title=None, cmin=None, cmax=None, lower_threshold=None, upper_threshold=None, target_plane=None, iso_energy_OSI_surface=None, interpolation=False, trajectory=None, all_trajectories=None, CTR_FR=None, exp_data=None, plot_ellipsoids=False, plot_exp_stems=False, colorbar_mode=True, legend_mode=False, axes_mode=False, elev=20, azim=120, figsize=(6.5, 6.5), ax=None, cax=None, savename=None, dpi=250, cmap='viridis'):
    # plot a publication-style 3d grid visualization with optional interpolated volume and colored trajectories
    
    # input
    # results is a dict with keys like 'Rm_{rm}_EL_{el}_wscale_{ws}' and per-key dict values
    # value_key is the scalar in results[key] to color the grid or interpolation
    # title is an optional title for the plot
    # cmin is the minimum value for colormap normalization
    # cmax is the maximum value for colormap normalization
    # lower_threshold is the lower threshold for filtering and interpolation range, defaults to global min of values if None
    # upper_threshold is the upper threshold for filtering and interpolation range, defaults to global max of values if None
    # target_plane enables plotting a closest-by-value plane, using value_key matching target_plane
    # iso_energy_OSI_surface enables plotting an iso-energy surface colored by OSI
    # interpolation can be False, 'surface', True, or 'volume'
    # trajectory is a single trajectory list of (Rm, EL, wscale_raw)
    # all_trajectories is a list of trajectories used to plot random examples
    # CTR_FR is a tuple of mean CTR and FR coordinates
    # exp_data is ((Rm_CTR, EL_CTR, wscale_CTR), (Rm_FR, EL_FR, wscale_FR))
    # plot_ellipsoids determines whether experimental ellipsoids are plotted
    # plot_exp_stems determines whether experimental stems are plotted
    # colorbar_mode determines whether a colorbar is plotted
    # legend_mode determines whether the legend is plotted
    # axes_mode determines whether thick axis triad lines are plotted
    # elev and azim control the camera angle
    # figsize sets the figure size for standalone plotting
    # ax assigns plot to an existing axes, if None a new figure and axes are created
    # cax is the colorbar axis
    # savename is the base filename to save pdf and png, if None no saving happens
    # dpi is the save resolution
    # cmap is the colormap used
    
    # output
    # fig is the matplotlib figure
    # ax is the matplotlib 3d axes
    
    standalone = ax is None
    fig = None
    if standalone:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # parse input data
    Rm, EL, wscale_raw, r_post, E_tot, values = parse_results_grid(results, value_key)
    w_syn_e = np.asarray(w_scale_to_w_e_syn(wscale_raw))

    # set color normalization bounds
    finite_vals = values[np.isfinite(values)]
    if cmin is None:
        cmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
    if cmax is None:
        cmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
    norm = Normalize(vmin=cmin, vmax=cmax, clip=True)

    # derive label for the colorbar
    value_label = value_key_text_plot(value_key, plot_mode='correlation_short')

    # set global default thresholds if not provided
    if finite_vals.size == 0:
        data_min, data_max = 0.0, 0.0
    else:
        data_min, data_max = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
    if lower_threshold is None:
        lower_threshold = data_min
    if upper_threshold is None:
        upper_threshold = data_max
    if lower_threshold > upper_threshold:
        lower_threshold, upper_threshold = upper_threshold, lower_threshold

    # classify grid points
    is_zero = (values == 0) | (~np.isfinite(values))
    is_below = (~is_zero) & (values < lower_threshold)
    is_within = (~is_zero) & (values >= lower_threshold) & (values <= upper_threshold)
    is_above = (~is_zero) & (values > upper_threshold)

    # plot discrete grid points if no special plane and no interpolation
    s_grid = 14
    if (target_plane is None) and (iso_energy_OSI_surface is None) and (not interpolation):
        ax.scatter(Rm[is_within], EL[is_within], w_syn_e[is_within], c=values[is_within], cmap=cmap, norm=norm, s=s_grid, depthshade=False, linewidths=0.0, alpha=0.7)
        ax.scatter(Rm[is_zero], EL[is_zero], w_syn_e[is_zero], facecolors='none', edgecolors='black', s=s_grid, depthshade=False, linewidths=0.2, alpha=0.7)
        if np.any(is_below):
            ax.scatter(Rm[is_below], EL[is_below], w_syn_e[is_below], c=values[is_below], cmap=cmap, norm=norm, s=s_grid, depthshade=False, linewidths=0.0, alpha=0.10)
        if np.any(is_above):
            ax.scatter(Rm[is_above], EL[is_above], w_syn_e[is_above], c=values[is_above], cmap=cmap, norm=norm, s=s_grid, depthshade=False, linewidths=0.0, alpha=0.10)
        if colorbar_mode:
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.4, pad=0.25, aspect=20, cax=cax) if cax is not None else fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.4, pad=0.25, aspect=20)
            cbar.set_label(value_label)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # plot target plane as closest-by-value per (rm, el)
    if target_plane is not None:
        idx = build_index_by_RmEL(results, value_key)
        eq_Rm, eq_EL, eq_wsyn, eq_val = [], [], [], []
        for (rm, el), entries in idx.items():
            best, best_diff = None, np.inf
            for e in entries:
                vv = e.get('value', None)
                if vv is None or (not np.isfinite(vv)):
                    continue
                diff = abs(vv - target_plane)
                if diff < best_diff:
                    best_diff, best = diff, e
            if best is not None:
                eq_Rm.append(rm)
                eq_EL.append(el)
                eq_wsyn.append(w_scale_to_w_e_syn([best['wscale']])[0])
                eq_val.append(best['value'])
        eq_Rm, eq_EL, eq_wsyn, eq_val = np.array(eq_Rm), np.array(eq_EL), np.array(eq_wsyn), np.array(eq_val)
        ax.scatter(eq_Rm, eq_EL, eq_wsyn, c=eq_val, cmap=cmap, norm=norm, s=10, depthshade=False, linewidths=0.0, alpha=1.0)
        if colorbar_mode:
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.4, pad=0.25, aspect=20, cax=cax) if cax is not None else fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.4, pad=0.25, aspect=20)
            cbar.set_label(value_label)
            cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # plot plotly-like volume via stacked isosurfaces
    if interpolation is True:

        pts = np.column_stack((Rm[is_within], EL[is_within], w_syn_e[is_within]))
        vals = values[is_within]
        if pts.shape[0] >= 10:
            n_interp = 60
            Rm_lin = np.linspace(Rm.min(), Rm.max(), n_interp)
            EL_lin = np.linspace(EL.min(), EL.max(), n_interp)
            W_lin = np.linspace(w_syn_e.min(), w_syn_e.max(), n_interp)
            Rm_mesh, EL_mesh, W_mesh = np.meshgrid(Rm_lin, EL_lin, W_lin, indexing="ij")
            vol = griddata(pts, vals, (Rm_mesh, EL_mesh, W_mesh), method="linear")

            vol_min, vol_max = float(np.nanmin(vol)), float(np.nanmax(vol))
            fill_val = cmin - 10 * (cmax - cmin + 1e-9)
            vol_filled = np.nan_to_num(vol, nan=fill_val)

            iso_min = max(lower_threshold, vol_min)
            iso_max = min(upper_threshold, vol_max)

            if iso_min < iso_max:
                surface_count = 12
                opacity_total = 0.8 #0.9
                alpha_each = opacity_total / surface_count
                levels = np.linspace(iso_min, iso_max, surface_count)
                for lvl in levels:
                    plot_isosurface_from_volume(ax, vol_filled, Rm_lin, EL_lin, W_lin, lvl, norm, cmap=cmap, alpha=alpha_each)
                if colorbar_mode:
                    mappable = ScalarMappable(norm=norm, cmap=cmap)
                    mappable.set_array(vals)
                    cbar = fig.colorbar(mappable, ax=ax, shrink=0.4, pad=0.25, aspect=20, cax=cax) if cax is not None else fig.colorbar(mappable, ax=ax, shrink=0.4, pad=0.25, aspect=20)
                    cbar.set_label(value_label)
                    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # plot colored trajectories
    if all_trajectories is not None:
        idx_value = build_index_by_RmEL(results, value_key=value_key)
        example = random.sample(all_trajectories, min(13, len(all_trajectories)))
        for traj in example:
            add_colored_trajectory_3d(ax, traj, idx_value, norm, cmap, w_scale_to_w_e_syn, linewidth=2, alpha=0.85, show_nodes=True, node_size=4, node_alpha=0.95, node_edgecolor="none", node_linewidth=0.0)

    if trajectory is not None:
        idx_value = build_index_by_RmEL(results, value_key=value_key)
        add_colored_trajectory_3d(ax, trajectory, idx_value, norm, cmap, w_scale_to_w_e_syn, linewidth=4, alpha=0.95, show_nodes=True, node_size=14, node_alpha=0.95, node_edgecolor="white", node_linewidth=0.4)

    # plot ellipsoids before experimental dots so dots remain visible
    if plot_ellipsoids and exp_data is not None:
        exp_CTR, exp_FR = exp_data
        Rm_CTR, EL_CTR, ws_CTR = np.array(exp_CTR[0]), np.array(exp_CTR[1]), np.array(exp_CTR[2])
        Rm_FR, EL_FR, ws_FR = np.array(exp_FR[0]), np.array(exp_FR[1]), np.array(exp_FR[2])
        w_CTR = np.asarray(w_scale_to_w_e_syn(ws_CTR))
        w_FR = np.asarray(w_scale_to_w_e_syn(ws_FR))

        mean_CTR = np.array([Rm_CTR.mean(), EL_CTR.mean(), w_CTR.mean()])
        cov_CTR = np.cov(np.vstack([Rm_CTR, EL_CTR, w_CTR]))
        Xc, Yc, Zc = generate_ellipsoid_data(mean_CTR, cov_CTR, std_factor=2.0)
        Zc = np.where(Zc >= 0, Zc, np.nan)

        mean_FR = np.array([Rm_FR.mean(), EL_FR.mean(), w_FR.mean()])
        cov_FR = np.cov(np.vstack([Rm_FR, EL_FR, w_FR]))
        Xf, Yf, Zf = generate_ellipsoid_data(mean_FR, cov_FR, std_factor=2.0)
        Zf = np.where(Zf >= 0, Zf, np.nan)

        ax.plot_surface(Xc, Yc, Zc, linewidth=0, antialiased=True, alpha=0.15, color='k')
        ax.plot_surface(Xf, Yf, Zf, linewidth=0, antialiased=True, alpha=0.15, color='r')

    # plot experimental data last so it sits on top visually
    if exp_data is not None:
        exp_CTR, exp_FR = exp_data
        Rm_CTR, EL_CTR, ws_CTR = np.array(exp_CTR[0]), np.array(exp_CTR[1]), np.array(exp_CTR[2])
        Rm_FR, EL_FR, ws_FR = np.array(exp_FR[0]), np.array(exp_FR[1]), np.array(exp_FR[2])
        w_CTR = np.asarray(w_scale_to_w_e_syn(ws_CTR))
        w_FR = np.asarray(w_scale_to_w_e_syn(ws_FR))

        ax.scatter(Rm_CTR, EL_CTR, w_CTR, s=16, c='k', depthshade=False, label='CTR data')
        ax.scatter(Rm_FR, EL_FR, w_FR, s=16, c='r', depthshade=False, label='FR data')

        if plot_exp_stems:
            floor_z = float(np.nanmin(w_syn_e))
            for x, y, z in zip(Rm_CTR, EL_CTR, w_CTR):
                ax.plot([x, x], [y, y], [floor_z, z], linewidth=1.5, alpha=0.7, color='k')
            for x, y, z in zip(Rm_FR, EL_FR, w_FR):
                ax.plot([x, x], [y, y], [floor_z, z], linewidth=1.5, alpha=0.7, color='r')

        if legend_mode:
            ax.legend(frameon=False, loc='lower left')

    # set label spacing
    ax.tick_params(axis='x', pad=-4)
    ax.tick_params(axis='y', pad=-4)
    ax.tick_params(axis='z', pad=-2)

    ax.set_xlabel(r'R$_m$ (M$\Omega$)', labelpad=-6)
    ax.set_ylabel(r'V$_{rest}$ (mV)', labelpad=-6)
    ax.set_zlabel(r'$\langle w_{syn,e}\rangle$ (nS)', labelpad=-6)

    if title:
        ax.set_title(title)

    # hardcode ticks for publication
    ax.set_xticks([50, 100, 150])
    ax.set_yticks([-90, -70, -50])
    ax.set_zticks([0.1, 0.7, 1.3])

    # set white panes and white background
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # draw optional thick axis triad using current limits
    if axes_mode:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        z0, z1 = ax.get_zlim()
        ax.plot([x0, x1], [y0, y0], [z0, z0], color='k', linewidth=2.5)
        ax.plot([x0, x0], [y0, y1], [z0, z0], color='k', linewidth=2.5)
        ax.plot([x0, x0], [y0, y0], [z0, z1], color='k', linewidth=2.5)

    # set view
    ax.view_init(elev=elev, azim=azim)

    # save figure if requested
    if standalone and savename is not None:
        fig.savefig(f"../Figures/{savename}.pdf", bbox_inches="tight", dpi=dpi)
        fig.savefig(f"../Figures/{savename}.png", bbox_inches="tight", dpi=dpi)
    if standalone:
        plt.show()


def add_balls_to_3d_ax(ax3d, data_CTR_like, data_FR_like, colors=('#4a61e8', '#c41c5f', 'black'), size=180, edgecolor='white', lw=1.2, zorder=10):
    # add the CTR-like / FR-like points to an existing mpl 3D axis
    
    color_CTR_like, color_FR_like, _ = colors

    R_m_CTR_like, E_L_CTR_like, w_scale_CTR_like = data_CTR_like['R_m'], data_CTR_like['E_L'], data_CTR_like['w_scale']
    R_m_FR_like,  E_L_FR_like,  w_scale_FR_like  = data_FR_like['R_m'],  data_FR_like['E_L'],  data_FR_like['w_scale']

    w_e_CTR_like = float(w_scale_to_w_e_syn(w_scale_CTR_like))
    w_e_FR_like  = float(w_scale_to_w_e_syn(w_scale_FR_like))

    # "balls" (scatter markers with white edge; depthshade gives some 3D feel)
    ax3d.scatter([R_m_CTR_like], [E_L_CTR_like], [w_e_CTR_like], s=size, c=[color_CTR_like], edgecolors=edgecolor, linewidths=lw, depthshade=True, zorder=zorder)

    ax3d.scatter([R_m_FR_like], [E_L_FR_like], [w_e_FR_like], s=size, c=[color_FR_like], edgecolors=edgecolor, linewidths=lw, depthshade=True, zorder=zorder)

    return (R_m_CTR_like, E_L_CTR_like, w_e_CTR_like, R_m_FR_like,  E_L_FR_like,  w_e_FR_like)


def plot_2D_colored_surface(results, value_key, w_scale_constant, title, cmin=None, cmax=None, savename=None):
    # plot 2D surface for closest w_scale
    
    # input 
    # results is the dictionary of saved values
    # value_key is the quantity to display color coded
    # w_scale_constant is the value of w_scale to be displayed
    # title is the title of the plot 
    # cmin is the minimum value of the color code
    # cmax is the maximum value of the color code
    

    # output 
    # fig is the 3D plot

    # find the closest available w_scale
    available_w_scales = []
    for key in results:
        params = key.split('_')
        w_scale = float(params[5])
        available_w_scales.append(w_scale)
    w_scale_used = min(available_w_scales, key=lambda x: abs(x - w_scale_constant))
    print(f'Using closest available w_scale: {w_scale_used}')

    data = []
    for key in results:
        params = key.split('_')
        R_m = float(params[1])
        E_L = float(params[3])
        w_scale = float(params[5])
        
        if w_scale == w_scale_used:
            data.append({
                'R_m': R_m,
                'E_L': E_L,
                value_key: results[key][value_key] if results[key][value_key] is not None else 0
            })
    
    df = pd.DataFrame(data)

    # get value_key_text 
    value_key_text = value_key_text_plot(value_key, plot_mode='grid')
    
    # create pivot table to get the grid
    pivot_table = df.pivot(index='E_L', columns='R_m', values=value_key)
    
    # extract R_m, E_L, and values from the pivot table
    R_m_values = pivot_table.columns.values
    E_L_values = pivot_table.index.values
    Z = pivot_table.values
    
    # calculate default cmin and cmax if not provided
    if cmin is None:
        cmin = np.nanmin(Z[Z != 0])  # exclude zero values for cmin calculation
    if cmax is None:
        cmax = np.nanmax(Z[Z != 0])  # exclude zero values for cmax calculation
    
    # create a mask for zero values
    Z_non_zero = np.where(Z != 0, Z, np.nan)
    Z_zero = np.where((Z == 0) | np.isnan(Z), 0, np.nan) # Z == 0, 0, np.nan)
    
    fig = go.Figure()

    # plot zero values in white
    fig.add_trace(go.Heatmap(
        z=Z_zero,
        x=R_m_values,
        y=E_L_values,
        colorscale=[[0, 'white'], [1, 'white']],
        showscale=False,
        hovertemplate='R_m: %{x}<br>E_L: %{y}<br>' + value_key + ': 0<br>',
        zmin=0,
        zmax=1))
    
    # plot non-zero values with Viridis colorscale
    fig.add_trace(go.Heatmap(
        z=Z_non_zero,
        x=R_m_values,
        y=E_L_values,
        colorscale='Viridis',
        colorbar=dict(title=value_key_text), 
        hovertemplate='R_m: %{x}<br>E_L: %{y}<br>' + value_key + ': %{z}<br>',
        zmin=cmin,
        zmax=cmax))
    
    fig.update_layout(
        title=title + f' (w_scale = {w_scale_used})',
        xaxis_title='R_m (MOhm)',
        yaxis_title='E_L (mV)')
    
    fig.show()
    
    # save the plot as an HTML file
    if savename is not None:
        pio.write_html(fig, '../Figures/' + str(savename) + '.html')

    return fig

def plot_2D_mountains(results, value_key, w_scale_constant, title, cmin=None, cmax=None, savename=None):
    # plot 2D surface for closest w_scale as a mountain
    
    # input
    # results is the dictionary of saved values
    # value_key is the quantity to display color coded
    # w_scale_constant is the value of w_scale to be displayed
    # title is the title of the plot 
    # cmin is the minimum value of the color code
    # cmax is the maximum value of the color code

    # output 
    # fig is the 3D plot

    available_w_scales = []
    for key in results:
        params = key.split('_')
        w_scale = float(params[5])
        available_w_scales.append(w_scale)
    w_scale_used = min(available_w_scales, key=lambda x: abs(x - w_scale_constant))
    print(f'Using closest available w_scale: {w_scale_used}')
    
    # initialize and load data
    data = []
    for key in results:
        params = key.split('_')
        R_m = float(params[1])
        E_L = float(params[3])
        w_scale = float(params[5])
        
        if w_scale == w_scale_used:
            data.append({
                'R_m': R_m,
                'E_L': E_L,
                value_key: results[key][value_key] if results[key][value_key] is not None else 0
            })
    
    df = pd.DataFrame(data)
    
    # create pivot table to get the grid
    pivot_table = df.pivot(index='E_L', columns='R_m', values=value_key)
    
    # extract R_m, E_L, and values from the pivot table
    R_m_values = pivot_table.columns.values
    E_L_values = pivot_table.index.values
    Z = pivot_table.values
    
    # calculate default cmin and cmax if not provided
    if cmin is None:
        cmin = np.nanmin(Z[Z != 0])  # exclude zero values for cmin calculation
    if cmax is None:
        cmax = np.nanmax(Z[Z != 0])  # exclude zero values for cmax calculation
    
    fig = go.Figure()

    # get value_key_text 
    value_key_text = value_key_text_plot(value_key, plot_mode='grid')
    
    # plot the surface
    fig.add_trace(go.Surface(
        z=Z,
        x=R_m_values,
        y=E_L_values,
        colorscale='Viridis',
        cmin=cmin,
        cmax=cmax,
        colorbar=dict(title=value_key_text),
        hovertemplate='R_m: %{x}<br>E_L: %{y}<br>' + value_key + ': %{z}<br>'))
    
    fig.update_layout(
        scene=dict(
        xaxis=dict(
            title=dict(
                text='R<sub>m</sub> (MΩ)',  # Axis label
                font=dict(size=20)), tickfont=dict(size=15), backgroundcolor='white', gridcolor='lightgrey', showbackground=True),
        yaxis=dict(
            title=dict(
                text='V<sub>rest</sub> (mV)',  # Axis label
                font=dict(size=20)), tickfont=dict(size=15), backgroundcolor='white', gridcolor='lightgrey', showbackground=True),
        zaxis=dict(
            title=dict(
                text=f'{value_key_text}',  # Axis label
                font=dict(size=20)), tickfont=dict(size=15), backgroundcolor='white', gridcolor='lightgrey', showbackground=True)),
    #title=title,
    width=850,
    height=850,
    font=dict(
        family='CMU Serif',  # LaTeX-like font
        color='black'),
    legend=dict(
            yanchor='top',
            y=-0.1,
            xanchor='left',
            x=0.05))
    
    fig.show()
    
    # save the plot as an HTML file
    if savename is not None:
        pio.write_html(fig, '../Figures/' + str(savename) + '.html')
    
    return fig


############################ video plotting functions ############################

def create_rotation_video(fig, savename, duration=2, fps=30, trajectory=None, revealing=False, eye=dict(x=-2.2, y=2.2, z=0.8)):
    # create a rotating video of the 3D grid around the z-axis which optionally animates a dot along a trajectory with the option to reveal the trajectory progressively

    # input
    # fig is the the 3D figure to rotate created by plotly
    # savename is the desired file name
    # duration is the duration of the video in seconds
    # fps are the frames per second of the video
    # trajectory is a list of tuples of (R_m, E_L, w_scale) coordinates
    # revealing if provided makes the trajectory being revealed step-by-step
    # eye is the dictionary of the initial camera position

    # output
    # creates a GIF video file saved to disk in the "../Figures/" folder

    # set up frame saving path
    output_path = f"../Figures/{savename}_frames"
    os.makedirs(output_path, exist_ok=True)

    num_frames = duration * fps # total number of frames
    angles = np.linspace(0, 360 if trajectory is None else 180, num_frames, endpoint=False) # rotation angles (in degrees)

    # prepare trajectory if provided
    if trajectory is not None:
        traj_R_m, traj_E_L, traj_w_scale = zip(*trajectory)
        num_points = len(trajectory)
        dot_positions = [
            trajectory[min(int(i * (num_points - 1) / (num_frames - 1)), num_points - 1)]
            for i in range(num_frames)]
    
    else:
        dot_positions = [None] * num_frames

    # generate and save each frame
    frame_files = []
    for i, angle in enumerate(angles):
        fig_frame = deepcopy(fig)

        if trajectory is not None:
            visible_idx = min(int(i * (num_points - 1) / (num_frames - 1)), num_points - 1)
            dot_R_m, dot_E_L, dot_w_scale = dot_positions[i]

            if revealing:
                # reveal trajectory up to the current point
                if visible_idx > 0:
                    fig_frame.add_trace(go.Scatter3d(
                        x=traj_R_m[:visible_idx + 1],
                        y=traj_E_L[:visible_idx + 1],
                        z=w_scale_to_w_e_syn(traj_w_scale[:visible_idx + 1]), 
                        mode='lines',
                        line=dict(color='green', width=10),
                        name='Trajectory', 
                        showlegend=False))
                    
            else:
                # full trajectory visible from start
                fig_frame.add_trace(go.Scatter3d(
                    x=traj_R_m,
                    y=traj_E_L,
                    z=w_scale_to_w_e_syn(traj_w_scale),
                    mode='lines',
                    line=dict(color='green', width=10),
                    name='Trajectory', 
                    showlegend=False))
                
            # add the moving dot
            fig_frame.add_trace(go.Scatter3d(
                x=[dot_R_m],
                y=[dot_E_L],
                z=[w_scale_to_w_e_syn(dot_w_scale)],
                mode='markers',
                marker=dict(size=20, color='green'),
                name='Moving Dot', 
                showlegend=False))

        # rotate camera
        fig_frame.update_layout(scene_camera=dict(
            eye=dict(x=eye['x'] * np.cos(np.radians(angle)),
                     y=eye['y'] * np.sin(np.radians(angle)),
                     z=eye['z'])))


        # save frame
        frame_path = os.path.join(output_path, f"frame_{i:04d}.png")
        fig_frame.write_image(frame_path, width=1920, height=1080)
        frame_files.append(frame_path)

    # combine frames into a GIF
    video_path = f"../Figures/{savename}.gif"
    with imageio.get_writer(video_path, mode="I", duration=1 / fps) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)

    # clean up
    shutil.rmtree(output_path)
    print(f"Video saved at {video_path}")

"""
def create_video_iso_E_tot_OSI(results, E_tot_array, fps=3, savename='video_iso_E_tot_OSI', cmin=None, cmax=None, exp_data=None):
    # create a video by sequentially plotting iso-energy surfaces color-coded by OSI

    # input
    # results is the dictionary of simulation results
    # E_tot_array is an array of target iso-energy values
    # fps are the frames per second of the video
    # savename is the desired file name
    # cmin is the minimum value for color scale
    # cmax is the maximum value for color scale
    # exp_data is the experimental data points for CTR & FR

    # output
    # creates a GIF file saved to disk in the "../Figures/" folder

    output_path = f"../Figures/{savename}_frames"
    os.makedirs(output_path, exist_ok=True)

    frame_paths = []

    for i, E_val in enumerate(E_tot_array):
        print(f"Generating frame {i+1}/{len(E_tot_array)} for E_tot = {E_val:.2e}")

        fig = plot_interactive_3D(
            results=results,
            value_key='E_tot',
            title=f"Energy OSI (E_tot ≈ {E_val:.2e})",
            exp_data=exp_data, 
            iso_energy_OSI_surface=E_val,
            cmin=cmin,
            cmax=cmax,
            savename=None)  # do not save as HTML
    
        frame_path = os.path.join(output_path, f"frame_{i:03d}.png")
        write_image(fig, frame_path, width=1080, height=900)
        frame_paths.append(frame_path)

    # create GIF
    gif_path = f"../Figures/{savename}.gif"
    with imageio.get_writer(gif_path, mode="I", duration=1 / fps) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # delete temporary frame images
    shutil.rmtree(output_path)

    print(f"Video saved at: {gif_path}")
    """
    
def create_video_iso_E_tot_OSI_dynamic_exp(results, E_tot_array, fps=3, savename='video_iso_E_tot_OSI', cmin=None, cmax=None, original_exp_data=None, plot_exp_stems=False, circled_values=False, base_font_size=40, eye=dict(x=1.8, y=1.8, z=1.3)):
    # create a video that gradually reveals experimental data points as their corresponding grid point's E_tot exceeds the current iso-energy surface value
    
    # input
    # results is the dictionary of simulation results
    # E_tot_array is an array of target iso-energy values
    # fps are the frames per second of the video
    # savename is the desired file name
    # cmin is the minimum value for color scale
    # cmax is the maximum value for color scale
    # original_exp_data is the experimental data points for CTR & FR
    # plot_exp_stems plots vertical lines on experimental data to make them better visible 
    # circled_values colors exp_data with color of nearest grid dot
    # base_font_size is the base font size
    # eye is the dictionary of the initial camera position

    # output
    # creates a GIF file saved to disk in the "../Figures/" folder
    

    # 1. Map experimental data points to their closest grid point (CTR and FR separately)
    closest_points_CTR = find_closest_grid_points_to_exp_data(results, original_exp_data[0])
    closest_points_FR = find_closest_grid_points_to_exp_data(results, original_exp_data[1])

    # 2. Convert to list of items for easier iteration
    exp_points_CTR = list(closest_points_CTR.items())  # [(point, data), ...]
    exp_points_FR = list(closest_points_FR.items())

    # 3. Prepare frame directory
    output_path = f"../Figures/{savename}_frames"
    os.makedirs(output_path, exist_ok=True)
    frame_paths = []

    # 4. Initialize lists for visible points
    visible_CTR_points = [[], [], []]  # Rm, EL, w_scale
    visible_FR_points = [[], [], []]

    for i, E_val in enumerate(E_tot_array):
        print(f"Generating frame {i+1}/{len(E_tot_array)} for E_tot = {E_val:.2e}")

        # 5. Check CTR points
        for point, data in exp_points_CTR:
            grid_E_tot = data["grid_values"].get("E_tot", None)
            if grid_E_tot is not None and grid_E_tot > E_val and list(point) not in list(zip(*visible_CTR_points)):
                for j in range(3):
                    visible_CTR_points[j].append(point[j])

        # 6. Check FR points
        for point, data in exp_points_FR:
            grid_E_tot = data["grid_values"].get("E_tot", None)
            if grid_E_tot is not None and grid_E_tot > E_val and list(point) not in list(zip(*visible_FR_points)):
                for j in range(3):
                    visible_FR_points[j].append(point[j])

        # 7. Create exp_data for this frame
        filtered_exp_data = (visible_CTR_points, visible_FR_points)

        # 8. Plot the frame
        fig = plot_interactive_3D(
            results=results,
            value_key='OSI',
            title=f"Energy OSI (E_tot ≈ {E_val:.2e})",
            exp_data=filtered_exp_data,
            iso_energy_OSI_surface=E_val,
            cmin=cmin,
            cmax=cmax,
            plot_exp_stems=plot_exp_stems,
            circled_values=circled_values,
            base_font_size=base_font_size, 
            eye=eye,
            savename=None)

        # 9. Save PNG frame
        frame_path = os.path.join(output_path, f"frame_{i:03d}.png")
        write_image(fig, frame_path, width=1080, height=900)
        frame_paths.append(frame_path)

    # 10. Create GIF
    gif_path = f"../Figures/{savename}.gif"
    with imageio.get_writer(gif_path, mode="I", duration=1 / fps) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # 11. Cleanup
    shutil.rmtree(output_path)
    print(f"Video saved at: {gif_path}")
    


   
############################ data plotting functions ############################

def generate_ellipse_data(mean, cov, n_std=2):
    # generate the parameters for an ellipse corresponding to a given confidence interval (n_std)

    # input
    # mean is the mean of the data (center of the ellipse)
    # cov is the covariance matrix of the data
    # n_std is the number of standard deviations to determine the ellipse size, default is 2 for 95% confidence

    # output
    # width is the width of the ellipse
    # height is the height of the ellipse
    # theta is the angle of the ellipse in degrees
    
    # eigenvalues and eigenvectors of the covariance matrix
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # calculate the angle in degrees (for the rotation of the ellipse)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # width and height of the ellipse (scaled by the standard deviation factor)
    width, height = 2 * n_std * np.sqrt(vals)
    
    return width, height, theta


def plot_2D_data_correlations(x, y, x_label, y_label, description, savename=None):
    # plot 2D data correlations & ellipses

    # input
    # x is a tuple of an array or list with the values of the x-axis
    # y is a tuple of an array or list for the values of the y-axis
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # description is a string for the title of the plot
    # savename is an optional name to save figure

    # output
    # corr_coef_CTR, corr_coef_FR are the correlation coefficients of CTR & FR
    # p_value_CTR, p_value_FR are the p-values of CTR & FR

    # unpack data
    x_CTR = x[0]
    y_CTR = y[0]
    x_FR = x[1]
    y_FR = y[1]

    # calculate correlation coefficients & p-values
    corr_coef_CTR, p_value_CTR = pearsonr(x_CTR, y_CTR)
    corr_coef_FR, p_value_FR = pearsonr(x_FR, y_FR)

    plt.figure()
    sns.regplot(x=x_CTR, y=y_CTR, ci=None, scatter_kws={"s": 50, "alpha": 0.5},color='black', label = 'experimental CTR pairs')
    plt.scatter(np.mean(x_CTR),np.mean(y_CTR),s = 100,color='black',label='mean CTR')
    sns.regplot(x=x_FR, y=y_FR, ci=None, scatter_kws={"s": 50, "alpha": 0.5},color='red', label = 'experimental FR pairs')
    plt.scatter(np.mean(x_FR),np.mean(y_FR),s = 100,color='red',label='mean FR')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(description + f'\nCTR Pearson correlation: {corr_coef_CTR:.2f}, CTR p-value: {p_value_CTR:.2f}\nFR Pearson correlation: {corr_coef_FR:.2f}, FR p-value: {p_value_FR:.2f}')
    plt.tight_layout()
    if savename is not None:
        plt.savefig('../Figures/' + str(savename)+'.pdf')
    plt.show()
    
    return corr_coef_CTR, p_value_CTR, corr_coef_FR, p_value_FR

def plot_2D_data_ellipses(R_m, E_L, x_label, y_label, description, savename=None):
    # plots scatter & ellipses of data with fading opacity for CTR and FR data

    # R_m is a tuple of CTR & FR membrane resistance data
    # E_L is a tuple of CTR & FR resting potential data
    # x_label is a string for the description of the x-axis
    # y_label is a string for the description of the y-axis
    # description is a string for the title of the plot
    # savename is an optional name to save the file
    
    # unpack data
    R_m_CTR = R_m[0]
    E_L_CTR = E_L[0]
    R_m_FR = R_m[1]
    E_L_FR = E_L[1]
    
    fig, ax = plt.subplots()
    
    # calculate mean and covariance for CTR and FR
    mean_CTR = np.array([np.mean(R_m_CTR), np.mean(E_L_CTR)])
    cov_CTR = np.cov(R_m_CTR, E_L_CTR)
    
    mean_FR = np.array([np.mean(R_m_FR), np.mean(E_L_FR)])
    cov_FR = np.cov(R_m_FR, E_L_FR)

    # generate ellipses with fading effect for CTR and FR
    width_CTR, height_CTR, theta_CTR = generate_ellipse_data(mean_CTR, cov_CTR, n_std=2)
    width_FR, height_FR, theta_FR = generate_ellipse_data(mean_FR, cov_FR, n_std=2)
    
    # plot fading ellipses for CTR
    for alpha in np.linspace(0.05, 0.25, 100):
        ellipse_CTR = Ellipse(xy=mean_CTR, width=width_CTR, height=height_CTR, angle=theta_CTR, edgecolor='none', facecolor='black', alpha=alpha)
        ax.add_patch(ellipse_CTR)
        width_CTR *= 0.95
        height_CTR *= 0.95

    # Plot fading ellipses for FR
    for alpha in np.linspace(0.05, 0.25, 100):
        ellipse_FR = Ellipse(xy=mean_FR, width=width_FR, height=height_FR, angle=theta_FR, edgecolor='none', facecolor='red', alpha=alpha)
        ax.add_patch(ellipse_FR)
        width_FR *= 0.95
        height_FR *= 0.95
    
    # scatter plot for the CTR and FR data points
    ax.scatter(R_m_CTR, E_L_CTR, color='black', label='CTR data')
    ax.scatter(R_m_FR, E_L_FR, color='red', label='FR data')

    # axis labels and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(description)
    ax.legend(frameon=False)

    plt.tight_layout()

    # save the plot if savename is provided
    if savename is not None:
        plt.savefig(f'../Figures/{savename}.pdf')

    plt.show()

def plot_3d_data_ellipsoids(R_m, E_L, z, z_data_mode, savename=None):
    # plot 3D scatter plot with ellipsoid for CTR and FR data, restricting ellipsoid to positive r_post values

    # input
    # R_m is a tuple of CTR & FR membrane resistance data
    # E_L is a tuple of CTR & FR resting potential data
    # z is a tuple of CTR & FR either postsynaptic firing rate or w_scale values
    # z_data_mode is a string which specifies the z input (either r_post or w_scale)
    # savename is an optional name to save the file

    # unpack data
    R_m_CTR = R_m[0]
    E_L_CTR = E_L[0]
    z_data_CTR = z[0]
    R_m_FR = R_m[1]
    E_L_FR = E_L[1]
    z_data_FR = z[1]
    
    # mean and covariance calculation for CTR
    mean_CTR = np.array([np.mean(R_m_CTR), np.mean(E_L_CTR), np.mean(z_data_CTR)])
    cov_CTR = np.cov(np.vstack((R_m_CTR, E_L_CTR, z_data_CTR)))
    
    # mean and covariance calculation for FR
    mean_FR = np.array([np.mean(R_m_FR), np.mean(E_L_FR), np.mean(z_data_FR)])
    cov_FR = np.cov(np.vstack((R_m_FR, E_L_FR, z_data_FR)))

    # generate ellipsoid data with a restriction to positive r_post and 2 std scaling
    x_CTR, y_CTR, z_CTR = generate_ellipsoid_data(mean_CTR, cov_CTR, std_factor=2)
    x_FR, y_FR, z_FR = generate_ellipsoid_data(mean_FR, cov_FR, std_factor=2)

    # apply mask to restrict ellipsoid to positive z values (z >= 0)
    mask_CTR = z_CTR >= 0
    x_CTR = np.where(mask_CTR, x_CTR, np.nan)
    y_CTR = np.where(mask_CTR, y_CTR, np.nan)
    z_CTR = np.where(mask_CTR, z_CTR, np.nan)

    mask_FR = z_FR >= 0
    x_FR = np.where(mask_FR, x_FR, np.nan)
    y_FR = np.where(mask_FR, y_FR, np.nan)
    z_FR = np.where(mask_FR, z_FR, np.nan)

    # create scatter plot for CTR data
    scatter_CTR = go.Scatter3d(
        x=R_m_CTR,
        y=E_L_CTR,
        z=z_data_CTR,
        mode='markers',
        marker=dict(size=5, color='black'),
        name='CTR data')

    # create scatter plot for FR data
    scatter_FR = go.Scatter3d(
        x=R_m_FR,
        y=E_L_FR,
        z=z_data_FR,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='FR data')

    # create 3D surface for CTR ellipsoid
    ellipsoid_CTR = go.Surface(
        x=x_CTR, y=y_CTR, z=z_CTR,
        colorscale=[[0, 'rgba(0, 0, 0, 0.2)'], [1, 'rgba(0, 0, 0, 0.2)']],
        showscale=False,
        name='CTR ellipsoid')

    # create 3D surface for FR ellipsoid
    ellipsoid_FR = go.Surface(
        x=x_FR, y=y_FR, z=z_FR,
        colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(255, 0, 0, 0.2)']],
        showscale=False,
        name='FR ellipsoid')

    # combine plots
    fig = go.Figure(data=[scatter_CTR, scatter_FR, ellipsoid_CTR, ellipsoid_FR])

    # update layout for the 3D plot
    if z_data_mode == 'r_post':
        fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="R<sub>m</sub> (MΩ)",  # Axis label
                    font=dict(size=20)), tickfont=dict(size=15), backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            yaxis=dict(
                title=dict(
                    text="V<sub>rest</sub> (mV)",  # Axis label
                    font=dict(size=20)), tickfont=dict(size=15), backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            zaxis=dict(
                title=dict(
                    text="r<sub>post</sub> (Hz)",  # Axis label
                    font=dict(size=20)), tickfont=dict(size=15), backgroundcolor="white", gridcolor="lightgrey", showbackground=True)))
    
    if z_data_mode == 'w_scale':
        fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="R<sub>m</sub> (MΩ)",  # Axis label
                    font=dict(size=20)), tickfont=dict(size=15), backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            yaxis=dict(
                title=dict(
                    text="V<sub>rest</sub> (mV)",  # Axis label
                    font=dict(size=20)), tickfont=dict(size=15), backgroundcolor="white", gridcolor="lightgrey", showbackground=True),
            zaxis=dict(
                title=dict(
                    text="w<sub>scale</sub>",  # Axis label
                    font=dict(size=20)), tickfont=dict(size=15), backgroundcolor="white", gridcolor="lightgrey", showbackground=True)))
    
    fig.update_layout(
    title="Cell pairs with 3D Gaussian ellipsoids (2 std)",
    width=900,
    height=700,
    font=dict(
        family="CMU Serif",  # LaTeX-like font
        color="black"),
    legend=dict(
            yanchor="top",
            y=-0.1,
            xanchor="left",
            x=0.05))

    fig.show()
    
    # save plot as an HTML file
    if savename is not None:
        pio.write_html(fig, '../Figures/' + str(savename) + '.html')
    


############################ information calculation plotting functions ############################

def plot_bin_sizes(binning_results, info_measure, figsize=(6, 5), ax=None, savename_mode=False):  #12, 8
    # plots heatmap of the different binning sizes
   
    # input
    # binning_results are the binning results
    # info_measure decides if MI or TE is plotted
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename_mode is an optional mode to save figure
   
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
       fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # retrieve data
    df_binning_results = pd.DataFrame(binning_results, columns=['bin_time', 'bins_rate', 'mi_bin', 'te_bin'])  # create data frame for plotting
    df_binning_results.to_csv('gridsearch_optimal_binning_results.csv', index=False)  # save data frame as CSV
    
    if info_measure == 'MI':
        info_pivot = df_binning_results.pivot(index='bin_time', columns='bins_rate', values='mi_bin')
        label = '$MI$ (bits)'
        description = '$MI$ bin size heatmap'
        print(f"Optimal {info_measure} combination: {df_binning_results.loc[df_binning_results['mi_bin'].idxmax()]}")
        
    if info_measure == 'TE':
        info_pivot = df_binning_results.pivot(index='bin_time', columns='bins_rate', values='te_bin')
        label = '$TE$ (bits)'
        description = '$TE$ bin size heatmap'
        print(f"Optimal {info_measure} combination: {df_binning_results.loc[df_binning_results['te_bin'].idxmax()]}")
        
        
    # main plotting part
    sns.heatmap(info_pivot, ax=ax, cmap='viridis', cbar_kws={'label': label})
    ax.set_xlabel('number of bins for rate')
    ax.set_ylabel('bin time (ms)')
    ax.set_title(description)
    
    # control tickslabels
    
    n_ticks = 3
    n_x = len(info_pivot.columns)
    x_idx = np.linspace(0, n_x - 1, min(n_ticks, n_x), dtype=int)  # indices of columns
    ax.set_xticks(x_idx + 0.5)  
    ax.set_xticklabels([info_pivot.columns[i] for i in x_idx], rotation=0)#, ha='right')

    n_y = len(info_pivot.index)
    y_idx = np.linspace(0, n_y - 1, min(n_ticks, n_y), dtype=int)
    ax.set_yticks(y_idx + 0.5)
    ax.set_yticklabels([info_pivot.index[i] for i in y_idx])

    mappable = ax.collections[0]          # QuadMesh from the heatmap
    cbar = mappable.colorbar              # attached colorbar
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks, min_n_ticks=n_ticks-1))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename_mode is True:
            if fig is None:
                fig = ax.figure
            path = f"../Figures/optimal_binsizes_{info_measure}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
    


############################ energy contribution plotting functions ############################


def plot_energy_stackplot(E_tot, E_HK, E_RP, E_AP, E_ST, E_glu, E_Ca, r_post, description, labels=['$E_{HK}$', '$E_{RP}$', '$E_{AP}$', '$E_{glu}$', '$E_{Ca^{2+}}$', '$E_{syn}$'], r_post_optimum=None, r_post_optimum_percentages=None, inverted=False, legend_pos=False, y_limit=None, y_label=True, color_r_post_optimum='white', figsize=(5,4), ax=None, savename=None):
    # plot stacked energy contributiors
    
    # input
    # E_tot is the total energy consumption in ATP/s
    # E_HK, E_RP, E_AP, E_ST, E_glu, E_Ca are the energy consumers of the postsynaptic neuron in ATP/s
    # r_post is the postsynaptic firing rate 
    # description is a string for the title of the plot
    # r_post_optimum is an optional argument to plot a dashed vertical line at r_post_optimum
    # r_post_optimum_percentages is an optional argument to plot percentages at r_post_optimum
    # inverted is an optional argument which inverts the x-axis
    # legend_pos is an optional argument to include legend and its position
    # y_limit is an optional float to define the limit of the y-axis
    # y_label includes label for y-axis
    # color_r_post_optimum sets the color for the optimum line
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    # create stacked plot
    ax.stackplot(
        r_post,
        E_HK,
        E_RP,
        E_AP,
        E_glu,
        E_Ca,
        E_ST,
        labels=labels,
        colors=['#808080', '#285F8B', '#D91A8F', '#017376', '#018F9B', '#018F76'])
    
    # plot formatting
    ax.set_xlabel('Mean firing rate (Hz)')
    ax.set_ylabel(y_label)
    ax.set_title(description)
    if legend_pos != False: 
        ax.legend(loc=legend_pos, reverse=True)
    #if legend_pos:
    #    h, l = ax.get_legend_handles_labels()
    #    ax.legend(h[::-1], l[::-1], loc=legend_pos)

    if y_limit != None: 
        ax.set_ylim(0, y_limit)
        
    # add a dotted line & percentages if r_post_optimum is provided
    if r_post_optimum is not None:
        # find the index closest to the optimal firing rate
        idx_optimum = (np.abs(r_post - r_post_optimum)).argmin()
        
        ax.plot([r_post_optimum, r_post_optimum], [0, E_tot[idx_optimum]], color=color_r_post_optimum, linestyle='--', linewidth=2.0, zorder=5)
        #ax.axvline(x=r_post_optimum, color=color_r_post_optimum, linestyle='--', linewidth=2.0)

        if r_post_optimum_percentages is not None:
        
            # calculate the energy contribution percentages
            E_HK_percent, E_RP_percent, E_AP_percent, E_ST_percent, E_glu_percent, E_Ca_percent = af.calculate_energy_contribution_percentages(E_tot, E_HK, E_RP, E_AP, E_ST, E_glu, E_Ca)
            
        
            # to display percentages at correct heights of optimal index use absolute values
            stack_values = np.array([E_HK[idx_optimum],
                                    E_RP[idx_optimum],
                                    E_AP[idx_optimum],
                                    E_glu[idx_optimum],
                                    E_Ca[idx_optimum],
                                    E_ST[idx_optimum]])
            
            percentages = np.array([E_HK_percent[idx_optimum],
                                    E_RP_percent[idx_optimum],
                                    E_AP_percent[idx_optimum],
                                    E_glu_percent[idx_optimum],
                                    E_Ca_percent[idx_optimum],
                                    E_ST_percent[idx_optimum]])
            
            cumulative = np.cumsum(stack_values)
    
            # place the text annotation in the middle of each segment
            for i, label in enumerate(['$E_{HK}$', '$E_{RP}$', '$E_{AP}$', '$E_{glu}$', '$E_{Ca^{2+}}$', '$E_{syn}$']):
                y_pos = cumulative[i] - stack_values[i] / 2
                
                if percentages[i] > 6:
                    if inverted is False:
                        ax.text(r_post_optimum + (r_post[-1] - r_post[0]) * 0.02, y_pos, f'{int(percentages[i])}%', color='white', ha='left', va='center')
                    if inverted is True:
                        ax.text(r_post_optimum + (r_post[-1] - r_post[0]) * 0.02, y_pos, f'{int(percentages[i])}%', color='white', ha='right', va='center') 
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # invert x-axis if the 'inverted' argument is set to True
    if inverted:
        ax.invert_xaxis()
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.spines['left'].set_visible(False)
        ax.set_xlim(r_post.max(), r_post.min())
        #plt.gca().invert_xaxis()
    else:
        ax.set_xlim(r_post.min(), r_post.max())
        
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
    
def plot_energy_pie_chart(E_HK, E_RP, E_AP, E_ST, E_glu, E_Ca, r_post, title_mode=True, label_mode='long', figsize=(5, 5), ax=None, savename=None):
    # plot pie chart of energy contributiors
    
    # input
    # E_HK, E_RP, E_AP, E_ST, E_glu, E_Ca are the energy consumers of the postsynaptic neuron in ATP/s
    # r_post is the postsynaptic firing rate 
    # title_mode decides if title is plotted or not
    # label_mode decides if long or short labels should be used 
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    """
    original_labels = [ 
        'House keeping',
        'Resting potential\n(reversal of Na$^{+}$)', 
        'Synaptic transmission\n(reversal of Na$^{+}$)',
        'Synaptic transmission\n(glutamate recycling)', 
        'Synaptic transmission\n(reversal of presyn Ca$^{2+}$)', 
        'Action potential\n(reversal of Na$^{+}$)']  # attention! changed order for plotting reasons!
    """
    if label_mode == 'long':
        # unicode labels for constant line heights
        original_labels = [ 
            'House keeping',
            'Resting potential\n(reversal of Na\u207A)', 
            'Synaptic transmission\n(reversal of Na\u207A)',
            'Synaptic transmission\n(glutamate recycling)', 
            'Synaptic transmission\n(reversal of presyn Ca\u00B2\u207A)', 
            'Action potential\n(reversal of Na\u207A)']  # attention! changed order for plotting reasons!
    
    if label_mode == 'short':
        original_labels = [ 
            '$E_{HK}$',
            '$E_{RP}$', 
            '$E_{syn}$',
            '$E_{glu}$', 
            '$E_{Ca^2+}$', 
            '$E_{AP}$']  # attention! changed order for plotting reasons!
        
    if label_mode == 'no':
        original_labels = [ 
            '',
            '', 
            '',
            '', 
            '', 
            '']  # attention! changed order for plotting reasons!
    colors = ['#808080', '#285F8B', '#018F76', '#017376', '#018F9B', '#D91A8F']
    original_sizes = [E_HK, E_RP, E_ST, E_glu, E_Ca, E_AP]
    
    E_tot = sum(original_sizes)
    
    # filter out entries where the percentage contribution is less than 1%
    filtered_labels = []
    filtered_sizes = []
    filtered_colors = []
    for label, size, color in zip(original_labels, original_sizes, colors):
        percentage = (size / E_tot) * 100 if E_tot > 0 else 0
        if percentage >= 1:
            filtered_labels.append(label)
            filtered_sizes.append(size)
            filtered_colors.append(color)
            
    # display percentage only if it's >= 1%
    def autopct_white(pct):
        return (r'%1.0f$\%%$' % pct) if pct >= 1 else ''  

    # create the pie chart
    wedges, texts, autotexts = ax.pie(filtered_sizes, labels=filtered_labels, autopct=autopct_white, startangle=90, colors=filtered_colors, wedgeprops={'edgecolor': 'black'}, labeldistance=1.2, pctdistance=0.76)
    
    # change the color of percentages to white
    for autotext in autotexts:
        autotext.set_color('white')

    ax.axis('equal')  # 'equal' aspect ratio ensures that pie is drawn as a circle
    if title_mode is True: 
        ax.set_title(f'ATP consumption of neuronal activity in mammal grey matter\n ($r_{{post}}$ = {round(r_post)} Hz)') 
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
    


############################ comparing energy budgets plotting functions ############################

def g_K_Attwell(V_RP, R_m, V_K, V_Na):
    # calculate K+ conductance according to Attwell & Laughlin 2002
    # input 
    # V_RP is the resting potential in V
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na are the reversal potentials for K+, Na+
    
    # output
    # g_K is the K+ conductance 

    g_K = (2 / R_m) * (V_RP - V_K) / (V_RP + 2*V_Na - 3*V_K)
    return g_K

def g_Na_Attwell(V_RP, R_m, V_K, V_Na):
    # calculate Na+ conductance according to Attwell & Laughlin 2002
    # input 
    # V_RP is the resting potential in V
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na are the reversal potentials for K+, Na+

    # output
    # g_Na is the Na+ conductance 

    g_Na = (3 / R_m) * (V_RP - V_K) / (V_RP + 2*V_Na - 3*V_K)
    return g_Na

def E_RP_Attwell(V_RP, R_m, V_K, V_Na, e):
    # calculate the resting potential energy according to Attwell & Laughlin 2002
    # input 
    # V_RP is the resting potential in V
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na are the reversal potentials for K+, Na+
    # e is the elementary constant 

    # output
    # E_RP is the resting potential energy consumption
    
    E_RP = (V_Na - V_RP) * (V_RP - V_K) / (e * R_m * (V_RP + 2*V_Na - 3*V_K))
    return E_RP
    
def g_K(V_RP, R_m, V_K, V_Na, V_h, alpha=0.05):
    # calculate K+ conductance
    # input 
    # V_RP is the resting potential in V
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # alpha is the ratio between g_Na and g_K

    # output
    # g_K is the K+ conductance 
    
    g_K = (9 / R_m) * ((V_RP - V_h) / (9 * (1 + alpha) * (V_RP - V_h) + 12 * (V_K - V_RP) + 8 * alpha * (V_Na - V_RP)))
    return g_K

def g_Na(V_RP, R_m, V_K, V_Na, V_h, alpha=0.05):
    # calculate K+ conductance
    # input 
    # V_RP is the resting potential in V
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # alpha is the ratio between g_Na and g_K

    # output
    # g_Na is the Na+ conductance 
    
    g_Na = (9 * alpha / R_m) * ((V_RP - V_h) / (9 * (1 + alpha) * (V_RP - V_h) + 12 * (V_K - V_RP) + 8 * alpha * (V_Na - V_RP)))
    return g_Na

def g_h(V_RP, R_m, V_K, V_Na, V_h, alpha=0.05):
    # calculate K+ conductance
    # input 
    # V_RP is the resting potential in V
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # alpha is the ratio between g_Na and g_K

    # output
    # g_h is the h-conductance 
    
    g_h = (4 / R_m) * ((3 * (V_K - V_RP) + 2 * alpha * (V_Na - V_RP)) / (9 * (1 + alpha) * (V_RP - V_h) + 12 * (V_K - V_RP) + 8 * alpha * (V_Na - V_RP)))
    return g_h
   
def E_RP(V_RP, R_m, V_K, V_Na, V_h, e, alpha=0.05):
    # calculate K+ conductance
    # input 
    # V_RP is the resting potential in V
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # e is the elementary constant 
    # alpha is the ratio between g_Na and g_K

    # output
    # E_RP is the resting potential energy consumption
    
    E_RP = (g_Na(V_RP, R_m, V_K, V_Na, V_h, alpha=0.05) * (V_Na - V_RP) / 3 + g_h(V_RP, R_m, V_K, V_Na, V_h, alpha=0.05) * (V_h - V_RP) / 4) / e
    return E_RP
    
def plot_conductances(R_m, alpha, V_K, V_Na, V_h, colors=['#57e7ff', '#9357ff', '#ff1d1d'], figsize=(8, 5), ax=None, savename=None):
    # plots conductances(V_RP)
    # input
    # R_m is the membrane resistance in Ohm
    # alpha is the ratio between g_Na and g_K
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # colors are the colors for color_K, color_Na, color_h
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure
    
    color_K, color_Na, color_h = colors[0], colors[1], colors[2]
    
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    V_RP_vec = np.linspace(-90e-3, -50e-3, 500)
    cond_vec = np.zeros((5, 500))

    for i, V_RP in enumerate(V_RP_vec):
        cond_vec[0, i] = g_K(V_RP, R_m, V_K, V_Na, V_h, alpha)
        cond_vec[1, i] = g_K_Attwell(V_RP, R_m, V_K, V_Na)
        cond_vec[2, i] = g_Na(V_RP, R_m, V_K, V_Na, V_h, alpha)
        cond_vec[3, i] = g_Na_Attwell(V_RP, R_m, V_K, V_Na)
        cond_vec[4, i] = g_h(V_RP, R_m, V_K, V_Na, V_h, alpha)

    ax.plot(1e3 * V_RP_vec, 1e9 * cond_vec[0], color_K, label='$g_{K,with\,h}$')
    ax.plot(1e3 * V_RP_vec, 1e9 * cond_vec[1], color_K, linestyle='dashed', label='$g_{K}$')#'$\n(no HCN)')
    ax.plot(1e3 * V_RP_vec, 1e9 * cond_vec[2], color_Na, label='$g_{Na,with\,h}$')
    ax.plot(1e3 * V_RP_vec, 1e9 * cond_vec[3], color_Na, linestyle='dashed', label='$g_{Na}$')#'\n(no HCN)')
    ax.plot(1e3 * V_RP_vec, 1e9 * cond_vec[4], color_h, label='$g_{h}$')
    ax.set_xlabel('$V_{RP}$ (mV)')
    ax.set_ylabel('conductance (nS)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1), frameon=True, facecolor='white', framealpha=1.0, edgecolor='none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def plot_energy_vs_V_RP(R_m, V_K, V_Na, V_h, alpha, ylim=None, figsize=(8, 5), ax=None, savename=None):
    # plots E_RP(V_RP)

    # input
    # R_m is the membrane resistance in Ohm
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # alpha is the ratio between g_Na and g_K
    # ylim decides if y-limits are used
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    e = 1.602e-19 # in C
    V_RP_vec = np.linspace(-90e-3, -50e-3, 500)
    E_RP_vec = np.array([E_RP(V_RP, R_m, V_K, V_Na, V_h, e, alpha) for V_RP in V_RP_vec])
    E_RP_Attwell_vec = np.array([E_RP_Attwell(V_RP, R_m, V_K, V_Na, e) for V_RP in V_RP_vec])

    ax.plot(1e3 * V_RP_vec, E_RP_vec/1e9, 'gray', label='HCN')
    ax.plot(1e3 * V_RP_vec, E_RP_Attwell_vec/1e9, 'gray', linestyle='dashed', label='no HCN')
    ax.set_ylim(ylim)
    ax.set_xlabel('$V_{RP}$ (mV)')
    ax.set_ylabel('$E_{RP}$ ($10^{9}$ATP/s)')
    #ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))

    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
        

def plot_energy_vs_R_m(V_RP, V_K, V_Na, V_h, alpha, ylim=None, figsize=(8, 5), ax=None, savename=None):
    # plots E_RP(R_m)

    # input
    # V_RP is the resting potential in V
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # alpha is the ratio between g_Na and g_K
    # ylim decides if y-limits are used
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    e = 1.602e-19 # in C
    R_m_vec = np.linspace(5e7, 2e8, 500)
    E_RP_vec = np.array([E_RP(V_RP, R_m, V_K, V_Na, V_h, e, alpha) for R_m in R_m_vec])
    E_RP_Attwell_vec = np.array([E_RP_Attwell(V_RP, R_m, V_K, V_Na, e) for R_m in R_m_vec])

    ax.plot(1e-6 * R_m_vec, E_RP_vec/1e9, 'gray', label='HCN')
    ax.plot(1e-6 * R_m_vec, E_RP_Attwell_vec/1e9, 'gray', linestyle='dashed', label='no HCN')
    ax.set_ylim(ylim)
    ax.set_xlabel('$R_{m}$ (M$\Omega$)')
    ax.set_ylabel('$E_{RP}$ ($10^{9}$ATP/s)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', frameon=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))

    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def compute_heatmap_data(V_K, V_Na, V_h, alpha):
    # computes 2D heat map data
    # input 
    # V_K, V_Na, V_h are the reversal potentials for K+, Na+, h-currents
    # alpha is the ratio between g_Na and g_K
    
    # output
    # V_RP_vec is an array of resting potential values in V
    # R_m_vec is an array of membrane resistance values in Ohm
    # E_RP_grid, E_RP_Attwell_grid are array for the energy values

    e = 1.602e-19 # in C
    V_RP_vec = np.linspace(-90e-3, -50e-3, 500)
    R_m_vec = np.linspace(5e7, 2e8, 500)

    E_RP_grid = np.array([[E_RP(V_RP, R_m, V_K, V_Na, V_h, e, alpha) for R_m in R_m_vec] for V_RP in V_RP_vec])
    E_RP_Attwell_grid = np.array([[E_RP_Attwell(V_RP, R_m, V_K, V_Na, e) for R_m in R_m_vec] for V_RP in V_RP_vec])
    return V_RP_vec, R_m_vec, E_RP_grid, E_RP_Attwell_grid

def plot_heatmap(data, CTR, FR, V_RP_vec, R_m_vec, title, label_CTR='CTR', label_FR='FR', cmap='copper', legend_mode=False, abs_scale=None, figsize=(8, 5), ax=None, savename=None):
    # plots heatmap

    # input
    # data is an array of energy data
    # CTR, FR are lists of control & food-restricted [R_m, E_L]
    # V_RP_vec is an array of resting potential values in V
    # R_m_vec is an array of membrane resistance values in Ohm
    # title is a string for the title of the subplot 
    # label_CTR, label_FR are strings for the CTR & FR points in the legend
    # cmap is a string for the colormap to be used
    # legend_mode decides whether to display a legend for CTR and FR markers
    # abs_scale is the scale of the color bar if provided   
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
    vmin = abs_scale[0] if abs_scale else None
    vmax = abs_scale[1] if abs_scale else None
    heatmap = ax.imshow(data/1e9, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    x_ticks = np.linspace(0, 499, 5)
    y_ticks = np.linspace(0, 499, 5)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    #ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.set_xticklabels([f'{v:.0f}' for v in np.linspace(R_m_vec[0]*1e-6, R_m_vec[-1]*1e-6, 5)])
    ax.set_yticklabels([f'{v:.0f}' for v in np.linspace(V_RP_vec[0]*1e3, V_RP_vec[-1]*1e3, 5)])
    ax.set_xlabel('$R_{m}$ (M$\Omega$)')
    ax.set_ylabel('$V_{RP}$ (mV)')
    X, Y = np.meshgrid(np.arange(500), np.arange(500))
    contours = ax.contour(X, Y, data, levels=8, colors='white', linewidths=0.8)
    CTR_tr = (np.interp(CTR[0], R_m_vec, np.arange(500)), np.interp(CTR[1], V_RP_vec, np.arange(500)))
    FR_tr = (np.interp(FR[0], R_m_vec, np.arange(500)), np.interp(FR[1], V_RP_vec, np.arange(500)))
    ax.plot(CTR_tr[0], CTR_tr[1], 'ko', markersize=6, label=label_CTR)
    ax.plot(FR_tr[0], FR_tr[1], 'ro', markersize=6, label=label_FR)
    # add an arrow connecting the points
    ax.annotate('', xy=FR_tr, xytext=CTR_tr, arrowprops=dict(facecolor='bisque', arrowstyle='->', lw=1.5))
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.7, pad=0.04)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
       
    cbar.set_label('$E_{RP}$ ($10^{9}$ATP/s)', rotation=270, labelpad=15)
    if legend_mode:
        ax.legend(loc='upper right') 

    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()


############################# plotting functions Zeldenrust #############################


def plot_hidden_state_input(t_stim, w_e_0, w_e_signal, r_e_signal, spike_times_e, T, tau_switch, savename=None):
    # plot hidden state, weighted synaptic input and convolved synaptic input

    # input
    # t_stim is the array with the binary hidden state (0 or 1) over time
    # w_e_0 is an array of length N_e with all normalized excitatory synaptic weights
    # w_e_signal is an array of length N_e_signal with normalized excitatory synaptic weights
    # r_e_signal is a list of arrays with firing rates for excitatory coding neurons in Hz
    # spike_times_e is a dictionary or list of spike times for excitatory neurons (optional for weighted calculation)
    # T is the simulation duration in ms
    # tau_switch is the time constant in ms defining hidden state switching probability
    # savename is an optional filename to save the figure

    # 1st row hidden state
    
    time = np.arange(T) / 1000.0  # ms to seconds

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax[0].step(time, t_stim, where='post', color='black', lw=1.5)
    ax[0].set_ylabel("hidden state (0/1)")
    ax[0].set_title(f"Hidden state and excitatory input (switching input, $\\tau_{{switch}} = {round(tau_switch,0)}\\ \\mathrm{{ms}}$)")
    ax[0].set_yticks([0, 1])
    ax[0].set_ylim(-0.1, 1.1)

    # 2nd row weighted synaptic input (sum of weights * firing rates)

    weighted_synaptic_input = np.sum(w_e_signal[:, np.newaxis] * r_e_signal, axis=0)

    ax[1].plot(time, weighted_synaptic_input, color='blue', lw=1)
    ax[1].set_ylabel("weighted synaptic input")

    # 3rd row convolved synaptic input (PSC approximation)

    total_excitatory_input = np.zeros(T)

    for neuron_idx in spike_times_e:
        spikes = spike_times_e[neuron_idx]
        spikes = (spikes / ms).astype(int)
        total_excitatory_input[spikes] += w_e_0[neuron_idx] # w_e_0 previously
        
    tau_kernel = 5  # ms
    kernel_size = 100  # ms window
    kernel = np.exp(-np.arange(kernel_size) / tau_kernel)
    kernel /= kernel.sum()

    convolved_input = np.convolve(total_excitatory_input, kernel, mode='same')

    ax[2].plot(time, convolved_input, color='green', lw=1)
    ax[2].set_ylabel("convolved spiking input")
    ax[2].set_xlabel("time / s")

    plt.tight_layout()

    if savename is not None:
        plt.savefig('../Figures/' + str(savename) + '.pdf')

    plt.show()
    
def plot_hidden_state_output(hidden_state, I_inj, V_m, spike_times_post, sampling_rate, savename=None):
    # plot hidden state, weighted synaptic input, convolved synaptic input, and postsynaptic spike times

    # input
    # hidden_state is an array with the binary hidden state (0 or 1) over time
    # I_inj is an array of injected current in pA
    # V_m is an array of membrane voltage in mV
    # spike_times_post is an array of postsynaptic spike times in ms
    # sampling_rate is the sampling rate in 1/ms
    # savename is an optional filename to save the figure

    # 1st row hidden state
    time = np.arange(len(hidden_state)) / 1000.0 / sampling_rate  # ms to seconds

    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    ax[0].step(time, hidden_state, where='post', color='black', lw=1.5)
    ax[0].set_ylabel("hidden state (0/1)")
    ax[0].set_title("Hidden state, voltage trace and neuronal output")
    ax[0].set_yticks([0, 1])
    ax[0].set_ylim(-0.1, 1.1)

    # 2nd row injected current
    ax[1].plot(time, I_inj, color='green', lw=1)
    ax[1].set_ylabel("I / pA")

    # 3rd row voltage trace
    ax[2].plot(time, V_m, color='blue', lw=1)
    ax[2].set_ylabel("V_m / mV")
    
    # 4th row postsynaptic spikes
    spike_times_post_sec = spike_times_post / 1000.0  # ms to seconds
    ax[3].vlines(spike_times_post_sec, ymin=0, ymax=1, color='red')
    ax[3].set_ylabel("postsynaptic spike times")
    ax[3].set_xlabel("time / s")
    ax[3].set_ylim(0, 1)

    plt.tight_layout()

    if savename is not None:
        plt.savefig('../Figures/' + str(savename) + '.pdf')

    plt.show()

def plot_hidden_state_input_output(t_stim, w_e_0, w_e_signal, r_e_signal, spike_times_e, T, tau_switch, spike_times_post, savename=None):
    # plot hidden state, weighted synaptic input, convolved synaptic input, and postsynaptic spike times

    # input
    # t_stim is the array with the binary hidden state (0 or 1) over time
    # w_e_0 is an array of length N_e with all normalized excitatory synaptic weights
    # w_e_signal is an array of length N_e_signal with normalized excitatory synaptic weights
    # r_e_signal is a list of arrays with firing rates for excitatory coding neurons in Hz
    # spike_times_e is a dictionary or list of spike times for excitatory neurons
    # T is the simulation duration in ms
    # tau_switch is the time constant in ms defining hidden state switching probability
    # spike_times_post is an array of postsynaptic spike times in ms
    # savename is an optional filename to save the figure

    # 1st row hidden state
    
    time = np.arange(T) / 1000.0  # ms to seconds

    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    ax[0].step(time, t_stim, where='post', color='black', lw=1.5)
    ax[0].set_ylabel("hidden state (0/1)")
    ax[0].set_title(f"Hidden state, excitatory input and neuronal output (switching input, $\\tau_{{switch}} = {round(tau_switch,0)}\\ \\mathrm{{ms}}$)")
    ax[0].set_yticks([0, 1])
    ax[0].set_ylim(-0.1, 1.1)

    # 2nd row weighted synaptic input (sum of weights * firing rates)

    weighted_synaptic_input = np.sum(w_e_signal[:, np.newaxis] * r_e_signal, axis=0)

    ax[1].plot(time, weighted_synaptic_input, color='blue', lw=1)
    ax[1].set_ylabel("weighted synaptic input")

    # 3rd row convolved synaptic input (PSC approximation)

    total_excitatory_input = np.zeros(T)

    for neuron_idx in spike_times_e:
        spikes = spike_times_e[neuron_idx]
        spikes = (spikes / ms).astype(int)
        total_excitatory_input[spikes] += w_e_0[neuron_idx]
        
    tau_kernel = 5  # ms
    kernel_size = 100  # ms window
    kernel = np.exp(-np.arange(kernel_size) / tau_kernel)
    kernel /= kernel.sum()

    convolved_input = np.convolve(total_excitatory_input, kernel, mode='same')

    ax[2].plot(time, convolved_input, color='green', lw=1)
    ax[2].set_ylabel("convolved spiking input")
    #ax[2].set_xlabel("time / s")

    # 4th row postsynaptic spikes
    spike_times_post_sec = spike_times_post / 1000.0  # ms to seconds
    ax[3].vlines(spike_times_post_sec, ymin=0, ymax=1, color='red')
    ax[3].set_ylabel("postsynaptic spikes")
    ax[3].set_xlabel("time / s")
    ax[3].set_ylim(0, 1)

    plt.tight_layout()

    if savename is not None:
        plt.savefig('../Figures/' + str(savename) + '.pdf')

    plt.show()

def plot_bin_sizes_Zeldenrust(binning_results): 
    # plot results of different bin sizes (time & rate space)
    
    # input
    # binning_results is a list of different binning results
    
    # prepare data for plotting
    df_binning_results = pd.DataFrame(binning_results, columns=['bin_time', 'bins_rate', 'mi_st_bin', 'mi_I_bin', 'mi_I_st_bin']) # create data frame for plotting
    df_binning_results.to_csv('gridsearch_optimal_binning_results.csv', index=False) # save data frame as CSV
    
    # plot bin sizes/numbers for MI spike train
    plt.figure(figsize=(12, 8))
    mi_pivot = df_binning_results.pivot(index='bin_time', columns='bins_rate', values='mi_st_bin')
    sns.heatmap(mi_pivot, cmap='viridis', cbar_kws={'label': 'Mutual information / bits'})
    plt.title('Mutual information (MI) of hidden state and spike train heatmap')
    plt.xlabel('number of bins for conductance')
    plt.ylabel('bin time / ms')
    plt.tight_layout()
    plt.savefig('../Figures/Zeldenrust_optimal_binsizes_MI_spike_train.pdf')
    plt.show()
    
    # plot bin sizes/numbers for MI input
    plt.figure(figsize=(12, 8))
    te_pivot = df_binning_results.pivot(index='bin_time', columns='bins_rate', values='mi_I_bin')
    sns.heatmap(te_pivot, cmap='viridis', cbar_kws={'label': 'Mutual information / bits'})
    plt.title('Mutual information (MI) of hidden state and input heatmap')
    plt.xlabel('number of bins for conductance')
    plt.ylabel('bin time / ms')
    plt.tight_layout()
    plt.savefig('../Figures/Zeldenrust_optimal_binsizes_MI_I.pdf')
    plt.show()

    # plot bin sizes/numbers for MI input spike train
    plt.figure(figsize=(12, 8))
    te_pivot = df_binning_results.pivot(index='bin_time', columns='bins_rate', values='mi_I_st_bin')
    sns.heatmap(te_pivot, cmap='viridis', cbar_kws={'label': 'Mutual information / bits'})
    plt.title('Mutual information (MI) of input and spike train heatmap')
    plt.xlabel('number of bins for conductance')
    plt.ylabel('bin time / ms')
    plt.tight_layout()
    plt.savefig('../Figures/Zeldenrust_optimal_binsizes_MI_I_spike_train.pdf')
    plt.show()
    
    # print optimal combination of bin size/number for MI and TE
    best_mi_st_bin = df_binning_results.loc[df_binning_results['mi_st_bin'].idxmax()]
    best_mi_I_bin = df_binning_results.loc[df_binning_results['mi_I_bin'].idxmax()]
    best_mi_I_st_bin = df_binning_results.loc[df_binning_results['mi_I_st_bin'].idxmax()]
    
    print('Optimal MI_spike_train combination:\n' + str(best_mi_st_bin))
    print('Optimal MI_I combination:\n' + str(best_mi_I_bin))
    print('Optimal MI_I_spike_train combination:\n' + str(best_mi_I_st_bin))


def plot_V_m_SBI_posterior_samples(t_ms, V_m_mean, V_m_samples, colors=["cadetblue", "blue"], scalebar_width=4.0, ylims=None, figsize=(5, 4), ax=None, savename=None):
    # plot mean V_m trace with ±1 std from sampled parameter traces as a shaded band
    
    # input
    # t_ms is the time vector in ms 
    # V_m_mean is the membrane potential trace for mean parameters in mV
    # V_m_samples are n_samples standard deviation traces in mV
    # colors are the colors
    # scalebar_width is the width of the scale bar
    # ylims sets the y-limits (ymin, ymax) with None keeps autoscale
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    # compute ±1 std envelope from samples
    if V_m_samples is not None and len(V_m_samples) > 0:
        V_m_std = np.std(V_m_samples, axis=0)
    else:
        V_m_std = np.zeros_like(V_m_mean)

    # main (mean) trace
    ax.plot(t_ms[:10000], V_m_mean[:10000], lw=1.0, alpha=1.0, color=colors[0])
        
    # shaded area here: ±1 std around the mean trace
    ax.fill_between(t_ms[:10000], V_m_mean[:10000] - 1*V_m_std[:10000], V_m_mean[:10000] + 1*V_m_std[:10000], alpha=0.4, color=colors[1], linewidth=1.0) 

    # alternatively plot every trace by itself: 
    #for trace in V_m_samples: 
        #ax.plot(t_ms[:10000], trace[:10000], lw=0.1, alpha=0.1, color='red')
        
    if ylims is not None:
        ax.set_ylim(ylims)
        ymin = ylims[0]
        ymax = ylims[1]
    else:
        ymin, ymax = ax.get_ylim()

    # remove frame/labels
    ax.axis('off')

    # compute limits after plotting (use autoscale if ylims not provided)
    yr = ymax - ymin

    # scalebar anchor (bottom-left-ish inside the panel)
    x0 = t_ms[0] - 12.0  
    y0 = ymin - 0.001 * yr

    scale_time_ms = 100  # time scalebar length in ms
    scale_time_dt = scale_time_ms  # same, but in ms on x-axis
    scale_V_m_mV = 15   # voltage scalebar length in mV

    # time scalebar
    ax.plot([x0, x0 + scale_time_dt], [y0, y0], lw=scalebar_width, color='black', clip_on=False, zorder=5)
    ax.text(x0 + scale_time_dt / 2.0, y0 - 0.04 * yr, f'{int(scale_time_ms)} ms', ha='center', va='top', clip_on=False)

    # voltage scalebar
    ax.plot([x0, x0], [y0, y0 + scale_V_m_mV], lw=scalebar_width, color='black', clip_on=False, zorder=5)
    ax.text(x0 - 40.0, y0 + scale_V_m_mV / 2.0, f'{int(scale_V_m_mV)} mV', ha='right', va='center', rotation=90, clip_on=False)

    # save & plot standalone figures
    if standalone is True:
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()


######################## correlation plotting functions Zeldenrust ########################

def value_key_text_plot_Zeldenrust(value_key, plot_mode):
    # get string for title or axis label for desired value_key

    # input
    # value_key is the quantity to display color coded
    # plot_mode decides whether label for grid or for correlation is returned

    # output
    # value_key_text is the respective text of the quantity to display color coded
    
    if plot_mode == 'grid_hover':
        if value_key in ('E_L', 'E_L_mV_list'):
            value_key_text = 'V<sub>rest</sub> (mV)'
        elif value_key in ('V_thresh', 'V_thresh_mV_list'):
            value_key_text = 'V<sub>thresh</sub> (mV)'
        elif value_key in ('R_m', 'R_m_mean_MOhm_list'):
            value_key_text = 'R<sub>m</sub> (MΩ)'
        elif value_key in ('C_m', 'C_m_mean_pF_list'):
            value_key_text = 'C<sub>m</sub> (pF)'
        elif value_key in ('Delta_T', 'Delta_T_mean_mV_list'):
            value_key_text = 'Δ<sub>T</sub> (mV)'
        elif value_key in ('V_reset', 'V_reset_mean_mV_list'):
            value_key_text = 'V<sub>reset</sub> (mV)'
        elif value_key in ('tau_w_ad', 'tau_w_mean_ms_list'):
            value_key_text = 'τ<sub>ad</sub> (ms)'
        elif value_key in ('a_ad', 'a_w_mean_nS_list'):
            value_key_text = 'a<sub>ad</sub> (nS)'
        elif value_key in ('b_ad', 'b_w_mean_nA_list'):
            value_key_text = 'b<sub>ad</sub> (nA)'
        elif value_key in ('r_post', 'firing_rate_Hz_list'):
            value_key_text = 'r<sub>post</sub> (Hz)'
        elif value_key in ('E_tot', 'E_tot_1e9_ATP_per_s_list'):
            value_key_text = 'E<sub>tot</sub> (10<sup>9</sup> ATP/s)'
        elif value_key in ('MI', 'MI_calculated_bits_list', 'MI_FZ_bits_list'):
            value_key_text = 'MI (bits)'
        elif value_key in ('MI_per_energy', 'MI_calculated_per_energy_list', 'MI_FZ_per_energy_list'):
            value_key_text = 'MI/E<sub>tot</sub> (bits/(10<sup>9</sup> ATP/s))'
        elif value_key in ('MICE', 'MICE_calculated_list', 'MICE_FZ_list'):
            value_key_text = 'CE<sub>MI</sub> (bits/Hz)'
        elif value_key in ('MICE_per_energy', 'MICE_calculated_per_energy_list', 'MICE_FZ_per_energy_list'):
            value_key_text = 'CE<sub>MI</sub>/E<sub>tot</sub> (bits/(Hz&#183;10<sup>9</sup> ATP/s))'
        elif value_key in ('CV_V_m',):
            value_key_text = 'CV<sub>V<sub>m</sub></sub>'
        elif value_key in ('CV_ISI',):
            value_key_text = 'CV<sub>ISI</sub>'
        elif value_key in ('CV_ISI_per_energy',):
            value_key_text = 'CV<sub>ISI</sub>/E<sub>tot</sub> (10<sup>-9</sup> s/ATP)'
        elif value_key in ('hit_fraction',):
            value_key_text = 'Hit fraction'
        elif value_key in ('false_alarm_fraction',):
            value_key_text = 'False-alarm fraction'
        elif value_key in ('pc1',):
            value_key_text = 'PC 1'
        elif value_key in ('pc2',):
            value_key_text = 'PC 2'
        elif value_key in ('pc3',):
            value_key_text = 'PC 3'
        else:
            value_key_text = f'{value_key}'
            
    elif plot_mode == 'grid_axis':
        if value_key in ('E_L', 'E_L_mV_list'):
            value_key_text = 'V<sub>rest</sub><br>(mV)'
        elif value_key in ('V_thresh', 'V_thresh_mV_list'):
            value_key_text = 'V<sub>thresh</sub><br>(mV)'
        elif value_key in ('R_m', 'R_m_mean_MOhm_list'):
            value_key_text = 'R<sub>m</sub><br>(MΩ)'
        elif value_key in ('C_m', 'C_m_mean_pF_list'):
            value_key_text = 'C<sub>m</sub><br>(pF)'
        elif value_key in ('Delta_T', 'Delta_T_mean_mV_list'):
            value_key_text = 'Δ<sub>T</sub><br>(mV)'
        elif value_key in ('V_reset', 'V_reset_mean_mV_list'):
            value_key_text = 'V<sub>reset</sub><br>(mV)'
        elif value_key in ('tau_w_ad', 'tau_w_mean_ms_list'):
            value_key_text = 'τ<sub>ad</sub><br>(ms)'
        elif value_key in ('a_ad', 'a_w_ad', 'a_w_mean_nS_list'):
            value_key_text = 'a<sub>ad</sub><br>(nS)'
        elif value_key in ('b_ad', 'b_w_ad', 'b_w_mean_nA_list'):
            value_key_text = 'b<sub>ad</sub><br>(nA)'
        elif value_key in ('r_post', 'firing_rate_Hz_list'):
            value_key_text = 'r<sub>post</sub><br>(Hz)'
        elif value_key in ('E_tot', 'E_tot_1e9_ATP_per_s_list'):
            value_key_text = 'E<sub>tot</sub><br>(10<sup>9</sup> ATP/s)'
        elif value_key in ('MI', 'MI_calculated_bits_list', 'MI_FZ_bits_list'):
            value_key_text = 'MI<br>(bits)'
        elif value_key in ('MI_per_energy', 'MI_calculated_per_energy_list', 'MI_FZ_per_energy_list'):
            value_key_text = 'MI/E<sub>tot</sub><br>(bits/(10<sup>9</sup> ATP/s))'
        elif value_key in ('MICE', 'MICE_calculated_list', 'MICE_FZ_list'):
            value_key_text = 'CE<sub>MI</sub><br>(bits/Hz)'
        elif value_key in ('MICE_per_energy', 'MICE_calculated_per_energy_list', 'MICE_FZ_per_energy_list'):
            value_key_text = 'CE<sub>MI</sub>/E<sub>tot</sub><br>(bits/(Hz&#183;10<sup>9</sup> ATP/s))'
        elif value_key in ('CV_V_m',):
            value_key_text = 'CV<sub>V<sub>m</sub></sub>'
        elif value_key in ('CV_ISI',):
            value_key_text = 'CV<sub>ISI</sub>'
        elif value_key in ('CV_ISI_per_energy',):
            value_key_text = 'CV<sub>ISI</sub>/E<sub>tot</sub><br>(10<sup>-9</sup> s/ATP)'
        elif value_key in ('hit_fraction',):
            value_key_text = 'Hit fraction'
        elif value_key in ('false_alarm_fraction',):
            value_key_text = 'False-alarm fraction'
        elif value_key in ('pc1',):
            value_key_text = 'PC 1'
        elif value_key in ('pc2',):
            value_key_text = 'PC 2'
        elif value_key in ('pc3',):
            value_key_text = 'PC 3'
        else:
            value_key_text = f'{value_key}'
            
    
    
    
    
    
    
    
    
    
    
    
    
    elif plot_mode == 'correlation': 
        if value_key in ('E_L', 'E_L_mV_list'):
            value_key_text = '$V_{rest}$ (mV)'
        elif value_key in ('r_post', 'firing_rate_Hz_list'):
            value_key_text = '$r_{post}$ (Hz)'
        elif value_key in ('V_thresh', 'V_thresh_mV_list'):
            value_key_text = '$V_{thresh}$ (mV)'
        elif value_key in ('R_m', 'R_m_mean_MOhm_list'):
            value_key_text = '$R_{m}$ (M$\\Omega$)'
        elif value_key in ('C_m', 'C_m_mean_pF_list'):
            value_key_text = '$C_{m}$ (pF)'
        elif value_key in ('Delta_T', 'Delta_T_mean_mV_list'): 
            value_key_text = '$\\Delta$T (mV)'
        elif value_key in ('V_reset', 'V_reset_mean_mV_list'):
            value_key_text = '$V_{reset}$ (mV)'
        elif value_key in ('tau_w_ad', 'tau_w_mean_ms_list'):
            value_key_text = '$\\tau_{ad}$ (ms)'
        elif value_key in ('a_ad', 'a_w_mean_nS_list'):
            value_key_text = '$a_{ad}$ (nS)'
        elif value_key in ('b_ad', 'b_w_mean_nA_list'):
            value_key_text = '$b_{ad}$ (nA)'
        elif value_key in ('MI', 'MI_calculated_bits_list', 'MI_FZ_bits_list'):
            value_key_text = '$MI$ (bits)'
        elif value_key in ('FI_calculated_list','FI_FZ_list'):
            value_key_text = '$FI$'
        elif value_key in ('E_tot', 'E_tot_1e9_ATP_per_s_list'):
            value_key_text = '$E_{tot}$ ($10^{9}$ ATP/s)'
        elif value_key in ('MICE', 'MICE_calculated_list', 'MICE_FZ_list'):
            value_key_text = '$CE_{MI}$ (bits/Hz)'
        elif value_key in ('MICE_per_energy', 'MICE_calculated_per_energy_list', 'MICE_FZ_per_energy_list'):
            value_key_text = '$MI/E_{tot}$ (bits/($10^{9}$ ATP/s))'
        elif value_key in ('MICE_calculated_per_energy_list', 'MICE_FZ_per_energy_list'):
            value_key_text = '$CE_{MI}/E_{tot}$ (bits/(Hz $10^{9}$ ATP/s))'
        elif value_key == 'I_syn_mean_pA_list':
            value_key_text = '$I_{syn}$ (pA)'
        elif value_key in ('CV_V_m',):
            value_key_text = '$CV_{V_{m}}$'
        elif value_key in ('CV_ISI',):
            value_key_text = '$CV_{ISI}'
        elif value_key in ('CV_ISI_per_energy',):
            value_key_text = '$CV_{ISI}/E{tot} ($10^{-9}$ s/ATP)'
        elif value_key in ('hit_fraction',):
            value_key_text = 'Hit fraction'
        elif value_key in ('false_alarm_fraction',):
            value_key_text = 'False-alarm fraction'
        elif value_key in ('pc1',):
            value_key_text = 'PC 1'
        elif value_key in ('pc2',):
            value_key_text = 'PC 2'
        elif value_key in ('pc3',):
            value_key_text = 'PC 3'
        else: 
            value_key_text = f'{value_key}'
            
    else:
        # fallback (e.g. correlation mode, or unknown)
        value_key_text = f'{value_key}'
        
    return value_key_text

def plot_correlation_exc_inh(x1, x2, y1, y2, x_label, y_label, z1=None, z2=None, z_label=None, x3=None, y3=None, z3=None, inverted_x=None, log_log=False, colors=['red', 'blue'], transparency=0.5,  figsize=(4,3), ax=None, savename=None): 
    # plot correlation between two variables with optional fit and color coding

    # input
    # x1 & x2 are arrays of x-values to be plotted (exc & inh)
    # y1 & y2 are arrays of y-values to be plotted (exc & inh)
    # x_label is a string to be used as x-axis label
    # y_label is a string to be used as y-axis label
    # z1 & z2 are arrays for color coding of scatter points
    # z_label is a string for the z colorbar label   
    # inverted_x is a boolean to invert the x-axis if desired
    # log_log s a boolean to enable loglog scaling if desired
    # color are the colors of exc & inh cells
    # transparency sets the transparency of experimental data points by parameter alpha 
    # savename is a string filename to save the figure as .pdf
    
    # output
    
    # prepare figure for standalone or assigned to axis
    standalone = ax is None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)#
    else:
        fig = ax.get_figure()
    
    # set up shared color normalization if z1 and/or z2 are given
    use_color = (z1 is not None) or (z2 is not None)
    norm = None
    cmap = None
    if use_color:
        z_all_list = []
        if z1 is not None:
            z_all_list.append(np.asarray(z1))
        if z2 is not None:
            z_all_list.append(np.asarray(z2))
        if len(z_all_list) > 0:
            z_all = np.concatenate(z_all_list)
            z_min = np.nanmin(z_all)
            z_max = np.nanmax(z_all)
            norm = Normalize(vmin=z_min, vmax=z_max)
            cmap = 'viridis'
            
    # create scatter plot of x & y data with optional color code z
    scatter = ax.scatter(x2, y2, s=50, alpha=transparency, label = 'inh cells', edgecolors=colors[1], linewidths=1.5, color=colors[1] if z2 is None else None, c=z2 if z2 is not None else None, cmap=cmap if z2 is not None else None, norm=norm if z2 is not None else None)
    scatter = ax.scatter(x1, y1, s=50, alpha=transparency, label = 'exc cells', edgecolors=colors[0], linewidths=1.5, color=colors[0] if z1 is None else None, c=z1 if z1 is not None else None, cmap=cmap if z2 is not None else None, norm=norm if z1 is not None else None)
    #scatter = ax.scatter(x2, y2, s=50, alpha=transparency, label = 'inh cells', edgecolors='blue', linewidths=1.5, color='blue' if z2 is None else None, c=z2 if z2 is not None else None, cmap=cmap if z2 is not None else None, norm=norm if z2 is not None else None)
    if x3 and y3 is not None: 
        scatter = ax.scatter(x3, y3, s=50, alpha=transparency, label = 'inh cells \nwith exc input', edgecolors='green', linewidths=1.5, color='green' if z3 is None else None, c=z3 if z3 is not None else None, cmap=cmap if z3 is not None else None, norm=norm if z3 is not None else None)
        
    # get correct axis labels
    x_label_text = value_key_text_plot_Zeldenrust(x_label, plot_mode='correlation')
    ax.set_xlabel(x_label_text)
    y_label_text = value_key_text_plot_Zeldenrust(y_label, plot_mode='correlation')
    ax.set_ylabel(y_label_text)
    if z_label is not None: 
        z_label_text = value_key_text_plot_Zeldenrust(z_label, plot_mode='correlation')
        #plt.colorbar(scatter, label=z_label_text) # add colorbar if z is provided
        cbar = fig.colorbar(scatter, ax=ax, label=z_label_text)
        ticks = [round(z_min, 2), round((z_min + z_max) / 2, 2), round(z_max, 2)]
        cbar.set_ticks(ticks)

    #plt.ylim(-0.05,0.69)
    if log_log == True: 
        plt.xscale('log')
        plt.yscale('log') 
    

    # add legend     
    # get current legend handles and labels
    auto_handles, auto_labels = ax.get_legend_handles_labels()
    if auto_handles != []:
        ax.legend(handles=auto_handles, labels=auto_labels, frameon=False)
        
    # invert x-axis if desired
    if inverted_x is not None:
        plt.gca().invert_xaxis()
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    
    # fix axes ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    
    
    # save & plot standalone figures
    if standalone is True:       
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()


# 3D plotting functions for grid search
def plot_Zeldenrust_interactive_3D(results, value_key, title, cmin=None, cmax=None, lower_threshold=None, upper_threshold=None, target_plane=None, interpolation=False, trajectory=None, all_trajectories=None, CTR_FR=None, exp_data=None, exp_data_value_key=None, plot_ellipsoids=False, colors=['red', 'blue'], savename=None):
    # input
    # results is the dictionary of saved values
    # value_key is the quantity to display color coded
    # title is the title of the plot 
    # cmin is the minimum value of the color code
    # cmax is the maximum value of the color code
    # lower_threshold is the value under which the points should be transparent
    # upper_threshold is the value over which the points should be transparent
    # target_plane is the value for a target energy plane (should be within the upper & lower threshold to be displayed correctly)
    # interpolation determines if the actual simulated data points should be exchanged by a contionous interpolated plane
    # trajectory is the trajectory connecting a starting point with an end point according to an energy-efficiency scheme
    # all_trajectories are 20 randomly chosen trajectories from all
    # CTR_FR are experimentally measured points to be displayed
    # exp_data are the experimental CTR & FR data points
    # exp_data_value_key is the quantity of the experimental data to display color coded
    # plot_ellipsoids determines if the ellipsoid should be plotted
    # colors are the color of exc & inh cells
    # savename is the name to save the grid as html file

    # output
    # fig is the 3D plot
    
    # initialize data
    R_m_values = []
    V_thresh_values = []
    tau_w_ad_values = []
    r_post_values = []
    E_tot_values = []
    values = []

    for key in results:
        params = key.split('_')
        R_m = float(params[1]) # get R_m value from key
        V_thresh = float(params[3]) # get V_thresh value from key
        tau_w_ad = float(params[5]) # get tau_w_ad value from key
        
        R_m_values.append(R_m)
        V_thresh_values.append(V_thresh)
        tau_w_ad_values.append(tau_w_ad)
        r_post_value = results[key]['r_post'] # access r_post value from dictionary
        r_post_values.append(r_post_value if r_post_value is not None else 0.0)  # replace None with 0
        E_tot_value = results[key]['E_tot'] # access E_tot value from dictionary
        E_tot_values.append(E_tot_value if E_tot_value is not None else 0.0)  # replace None with 0
        if value_key != 'MI_vec': 
            value = results[key][value_key] # access desired value from dictionary
        if value_key == 'MI_vec': 
            value = results[key][value_key][7] # access desired value from dictionary
        values.append(value if value is not None else 0.0) # replace None with 0
        
    # calculate default cmin and cmax if not provided
    if cmin is None:
        cmin = min([v for v in values if not np.isnan(v)])
    if cmax is None:
        cmax = max([v for v in values if not np.isnan(v)])

    # create separate lists for zero, below (non-zero) threshold, within thresholds, and above threshold values
    zero_indices = [i for i, v in enumerate(values) if v == 0 or v is None]
    below_threshold_indices = [i for i, v in enumerate(values) if v is not None and v != 0 and lower_threshold is not None and v < lower_threshold]
    within_threshold_indices = [i for i, v in enumerate(values) if v is not None and v != 0 and (lower_threshold is None or v >= lower_threshold) and (upper_threshold is None or v <= upper_threshold)]
    above_threshold_indices = [i for i, v in enumerate(values) if v is not None and v != 0 and upper_threshold is not None and v > upper_threshold]

    zero_R_m_values = [R_m_values[i] for i in zero_indices]
    zero_V_thresh_values = [V_thresh_values[i] for i in zero_indices]
    zero_tau_w_ad_values = [tau_w_ad_values[i] for i in zero_indices]
    zero_r_post_values = [r_post_values[i] for i in zero_indices]
    zero_E_tot_values = [E_tot_values[i] for i in zero_indices]
    
    below_threshold_R_m_values = [R_m_values[i] for i in below_threshold_indices]
    below_threshold_V_thresh_values = [V_thresh_values[i] for i in below_threshold_indices]
    below_threshold_tau_w_ad_values = [tau_w_ad_values[i] for i in below_threshold_indices]
    below_threshold_r_post_values = [r_post_values[i] for i in below_threshold_indices]
    below_threshold_E_tot_values = [E_tot_values[i] for i in below_threshold_indices]
    below_threshold_values = [values[i] for i in below_threshold_indices]

    within_threshold_R_m_values = [R_m_values[i] for i in within_threshold_indices]
    within_threshold_V_thresh_values = [V_thresh_values[i] for i in within_threshold_indices]
    within_threshold_tau_w_ad_values = [tau_w_ad_values[i] for i in within_threshold_indices]
    within_threshold_r_post_values = [r_post_values[i] for i in within_threshold_indices]
    within_threshold_E_tot_values = [E_tot_values[i] for i in within_threshold_indices]
    within_threshold_values = [values[i] for i in within_threshold_indices]

    above_threshold_R_m_values = [R_m_values[i] for i in above_threshold_indices]
    above_threshold_V_thresh_values = [V_thresh_values[i] for i in above_threshold_indices]
    above_threshold_tau_w_ad_values = [tau_w_ad_values[i] for i in above_threshold_indices]
    above_threshold_r_post_values = [r_post_values[i] for i in above_threshold_indices]
    above_threshold_E_tot_values = [E_tot_values[i] for i in above_threshold_indices]
    above_threshold_values = [values[i] for i in above_threshold_indices]
        
    fig = go.Figure()
    
    value_key_text = value_key_text_plot(value_key, plot_mode='grid')
    
    # find the target plane points (e.g. equi-energy plane) if desired
    if target_plane is not None:
        equi_R_m_values = []
        equi_V_thresh_values = []
        equi_tau_w_ad_values = []
        equi_values = []
        for r_m, v_thresh in zip(R_m_values, V_thresh_values):
            min_diff = float('inf')
            closest_tau_w_ad = None
            for key in results:
                params = key.split('_')
                R_m = float(params[1])
                V_thresh = float(params[3])
                tau_w_ad = float(params[5])
                if R_m == r_m and V_thresh == v_thresh:
                    value = results[key][value_key]
                    if value is not None:
                        diff = abs(value - target_plane)
                        if diff < min_diff:
                            min_diff = diff
                            closest_tau_w_ad = tau_w_ad
                            closest_value = value
            if closest_tau_w_ad is not None:
                equi_R_m_values.append(r_m)
                equi_V_thresh_values.append(v_thresh)
                equi_tau_w_ad_values.append(closest_tau_w_ad)
                equi_values.append(closest_value)
        
        # print lower and upper values based on the target plane values
        equi_min = min(equi_values)
        equi_max = max(equi_values)
        
        print(f'Lower value: {equi_min}')
        print(f'Upper value: {equi_max}')
    
        # plot equi-value plane
        fig.add_trace(go.Scatter3d(
            x=equi_R_m_values,
            y=equi_V_thresh_values,
            z=equi_tau_w_ad_values,
            mode='markers',
            marker=dict(
                size=5,
                color=equi_values,
                colorscale='Viridis',
                colorbar=dict(title=dict(text=value_key_text,font=dict(size=20)),tickfont=dict(size=15)),
                cmin=cmin,
                cmax=cmax,
                opacity=1.0),
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>'))

    # plot interpolated plane instead of discrete points if enabled
    if interpolation:
        # define grid for interpolation
        R_m_grid = np.linspace(min(R_m_values), max(R_m_values), 30)
        V_thresh_grid = np.linspace(min(V_thresh_values), max(V_thresh_values), 30)
        tau_w_ad_grid = np.linspace(min(tau_w_ad_values), max(tau_w_ad_values), 30)
        
        # prepare meshgrid
        grid_R_m, grid_V_thresh, grid_tau_w_ad = np.meshgrid(R_m_grid, V_thresh_grid, tau_w_ad_grid)
        
        # interpolate using griddata
        interpolated_values = griddata((within_threshold_R_m_values, within_threshold_V_thresh_values, within_threshold_tau_w_ad_values), within_threshold_values, (grid_R_m, grid_V_thresh, grid_tau_w_ad), method='linear')
        
        # plot interpolated volume as isosurface
        fig.add_trace(go.Volume(
            x=grid_R_m.flatten(),
            y=grid_V_thresh.flatten(),
            z=grid_tau_w_ad.flatten(),
            value=interpolated_values.flatten(),
            isomin=lower_threshold,
            isomax=upper_threshold,
            opacity=0.1,
            surface_count=5,
            colorscale='Viridis',
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>',
            colorbar=dict(title=dict(text=value_key_text, font=dict(size=20)), tickfont=dict(size=15))))
        
    # plot and calculate grid if not target_plane or interpolation desired
    elif target_plane is None or interpolation is None:
        
        # plot within threshold values
        fig.add_trace(go.Scatter3d(
            x=within_threshold_R_m_values,
            y=within_threshold_V_thresh_values,
            z=within_threshold_tau_w_ad_values,
            customdata=np.column_stack((within_threshold_r_post_values, within_threshold_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color=within_threshold_values,
                colorscale='Viridis',
                colorbar=dict(title=dict(text=value_key_text,font=dict(size=20)),tickfont=dict(size=15)),
                cmin=cmin,
                cmax=cmax,
                opacity=1.0),
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            name='Within thresholds'))

        # plot zero values as empty circles
        fig.add_trace(go.Scatter3d(
            x=zero_R_m_values,
            y=zero_V_thresh_values,
            z=zero_tau_w_ad_values,
            customdata=np.column_stack((zero_r_post_values, zero_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color='rgba(0,0,0,0)',  # make them transparent
                line=dict(color='black', width=0.5)),  # black outline
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': 0',
            name='Zero values'))

        # plot below threshold values in color but with low opacity
        fig.add_trace(go.Scatter3d(
            x=below_threshold_R_m_values,
            y=below_threshold_V_thresh_values,
            z=below_threshold_tau_w_ad_values,
            customdata=np.column_stack((below_threshold_r_post_values, below_threshold_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color=below_threshold_values,
                colorscale='Viridis',
                colorbar=dict(title=dict(text=value_key_text,font=dict(size=20)),tickfont=dict(size=15)),
                cmin=cmin,
                cmax=cmax,
                opacity=0.1),
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            name='Below threshold values'))

        # plot above threshold values in color but with low opacity
        fig.add_trace(go.Scatter3d(
            x=above_threshold_R_m_values,
            y=above_threshold_V_thresh_values,
            z=above_threshold_tau_w_ad_values,
            customdata=np.column_stack((above_threshold_r_post_values, above_threshold_E_tot_values)),
            mode='markers',
            marker=dict(
                size=5,
                color=above_threshold_values,
                colorscale='Viridis',
                colorbar=dict(title=dict(text=value_key_text,font=dict(size=20)),tickfont=dict(size=15)),
                cmin=cmin,
                cmax=cmax,
                opacity=0.1),
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>r_post: %{customdata[0]:.2f}<br>E_tot: %{customdata[1]:.2f}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            name='Above threshold values'))

    # plot a single trajectory as a red line
    if trajectory is not None:
        traj_R_m, traj_V_thresh, traj_tau_w_ad = zip(*trajectory)
        fig.add_trace(go.Scatter3d(
            x=traj_R_m,
            y=traj_V_thresh,
            z=traj_tau_w_ad,
            mode='lines', # with markers: 'lines+markers'
            line=dict(color=colors[0], width=4),
            marker=dict(size=4, color=colors[0]),
            name='single trajectory'))
    
    # if all_trajectories is provided plot 20 random example trajectories 
    if all_trajectories is not None:
        # plot 20 random example trajectories from all_trajectories
        example_trajectories = random.sample(all_trajectories, 20)  # Select 20 random trajectories


        for traj in example_trajectories:
            traj_R_m, traj_V_thresh, traj_tau_w_ad = zip(*traj)
            traj_values = [results[f'Rm_{rm}_EL_{el}_wscale_{ws}'][value_key] for rm, el, ws in traj]
            fig.add_trace(go.Scatter3d(
                x=traj_R_m,
                y=traj_V_thresh,
                z=traj_tau_w_ad,
                mode='lines',
                line=dict(
                    color=traj_values,
                    colorscale='Viridis',
                    cmin=cmin,
                    cmax=cmax,
                    width=6),
                opacity=0.95,
                showlegend=False,
                name='Example Trajectories'))

        # find local maxima
        filtered_uniquV_threshocal_maxima, filtered_uniquV_threshocal_maxima_values = find_local_maxima(results, all_trajectories, value_key, non_zero_filtering=False)

        # check if there are any valid maxima left after filtering and plot them
        if filtered_uniquV_threshocal_maxima:
            # unpack the filtered local maxima values for plotting
            local_R_m, local_V_thresh, local_tau_w_ad = zip(*filtered_uniquV_threshocal_maxima)

            # plot filtered local maxima
            fig.add_trace(go.Scatter3d(
                x=local_R_m,
                y=local_V_thresh,
                z=local_tau_w_ad,
                mode='markers',
                marker=dict(
                    size=8,
                    color=filtered_uniquV_threshocal_maxima_values,  # use the filtered values for coloring
                    colorscale='Viridis',
                    cmin=cmin,
                    cmax=cmax,
                    opacity=1.0),
                hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
                name='Local Maxima'))
        
    # plot mean CTR in black and mean FR point in dark red
    if CTR_FR is not None:
        CTR_coords, FR_coords = CTR_FR
        CTR_R_m, CTR_V_thresh, CTR_tau_w_ad = CTR_coords
        FR_R_m, FR_V_thresh, FR_tau_w_ad = FR_coords
        
        fig.add_trace(go.Scatter3d(
            x=[CTR_R_m],
            y=[CTR_V_thresh],
            z=[CTR_tau_w_ad],
            mode='markers',
            marker=dict(
                size=8,
                color='black'),
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            name='CTR Point'))
        
        fig.add_trace(go.Scatter3d(
            x=[FR_R_m],
            y=[FR_V_thresh],
            z=[FR_tau_w_ad],
            mode='markers',
            marker=dict(
                size=8,
                color='darkred'
                ),
            hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>' + value_key + ': %{marker.color:.2f}<extra></extra>',
            name='FR Point'))

    # plot experimental data if provided
    if exp_data is not None:
        R_m_exp, V_thresh_exp, tau_w_ad_exp = exp_data['R_m_estimated_list'], exp_data['V_thresh_list'], exp_data['tau_w_mean_estimated_list']

        """
        # old and new ranges
        old_min, old_max = -58, -7
        new_min, new_max = -63, -58
        # scale threshold coordinates
        V_thresh_exp = new_min + (np.array(V_thresh_exp) - old_min) * (new_max - new_min) / (old_max - old_min)
        """

        if exp_data_value_key is not None: 
            
            z_exp = exp_data[exp_data_value_key]

            if exp_data_value_key != 'tau_switching':
                fig.add_trace(go.Scatter3d(x=R_m_exp, y=V_thresh_exp, z=tau_w_ad_exp, mode='markers', marker=dict(size=5, color=z_exp, colorscale='Viridis', colorbar=dict(title=dict(text=f'{exp_data_value_key}', font=dict(size=20)), tickfont=dict(size=15)), cmin=cmin, cmax=cmax, opacity=1.0), hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>', name='Exp data'))
            
            if exp_data_value_key == 'tau_switching':
                
                # initialize lists for 50 and 250
                lists_250 = {'tau_switching_list': [], 'R_m_estimated_list': [], 'V_thresh_list': [], 'tau_w_mean_estimated_list': []}
                lists_50  = {'tau_switching_list': [], 'R_m_estimated_list': [], 'V_thresh_list': [], 'tau_w_mean_estimated_list': []}

                for i in range(len(z_exp)):
                    target = lists_250 if z_exp[i] == 250 else lists_50
                    target['tau_switching_list'].append(z_exp[i])
                    target['R_m_estimated_list'].append(R_m_exp[i])
                    target['tau_w_mean_estimated_list'].append(tau_w_ad_exp[i])
                    target['V_thresh_list'].append(V_thresh_exp[i])

                # access the results:
                #tau_switching_list_50 = lists_50['tau_switching_list']
                R_m_estimated_list_50 = lists_50['R_m_estimated_list']
                tau_w_mean_estimated_list_50 = lists_50['tau_w_mean_estimated_list']
                V_thresh_list_50 = lists_50['V_thresh_list']
                #tau_switching_list_250 = lists_250['tau_switching_list']
                R_m_estimated_list_250 = lists_250['R_m_estimated_list']
                tau_w_mean_estimated_list_250 = lists_250['tau_w_mean_estimated_list']
                V_thresh_list_250 = lists_250['V_thresh_list']

                fig.add_trace(go.Scatter3d(x=R_m_estimated_list_50, y=V_thresh_list_50, z=tau_w_mean_estimated_list_50, mode='markers', marker=dict(size=5, color='blue'), hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>', name='Inh data'))
                fig.add_trace(go.Scatter3d(x=R_m_estimated_list_250, y=V_thresh_list_250, z=tau_w_mean_estimated_list_250, mode='markers', marker=dict(size=5, color='red'), hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>', name='Exc data'))
        
        else: 
            # plot CTR and FR scatter points
            fig.add_trace(go.Scatter3d(x=R_m_exp, y=V_thresh_exp, z=tau_w_ad_exp, mode='markers', marker=dict(size=5, color='black'), hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>', name='Exp data'))
        
        
    # plot ellipsoid if desired
    if plot_ellipsoids and exp_data is not None:
        exp_data_CTR = exp_data[0]
        exp_data_FR = exp_data[1]
        R_m_CTR, V_thresh_CTR, tau_w_ad_CTR = exp_data_CTR[0], exp_data_CTR[1], exp_data_CTR[2]
        R_m_FR, V_thresh_FR, tau_w_ad_FR = exp_data_FR[0], exp_data_FR[1], exp_data_FR[2]
        
        # get CTR ellipsoid parameters
        mean_CTR = np.array([np.mean(R_m_CTR), np.mean(V_thresh_CTR), np.mean(tau_w_ad_CTR)])
        cov_CTR = np.cov(np.vstack((R_m_CTR, V_thresh_CTR, tau_w_ad_CTR)))
        x_CTR, y_CTR, z_CTR = generate_ellipsoid_data(mean_CTR, cov_CTR, std_factor=2)
        mask_CTR = z_CTR >= 0
        x_CTR, y_CTR, z_CTR = [np.where(mask_CTR, axis, np.nan) for axis in (x_CTR, y_CTR, z_CTR)]
        
        # get FR ellipsoid parameters
        mean_FR = np.array([np.mean(R_m_FR), np.mean(V_thresh_FR), np.mean(tau_w_ad_FR)])
        cov_FR = np.cov(np.vstack((R_m_FR, V_thresh_FR, tau_w_ad_FR)))
        x_FR, y_FR, z_FR = generate_ellipsoid_data(mean_FR, cov_FR, std_factor=2)
        mask_FR = z_FR >= 0
        x_FR, y_FR, z_FR = [np.where(mask_FR, axis, np.nan) for axis in (x_FR, y_FR, z_FR)]
             
        # plot CTR and FR ellipsoids
        fig.add_trace(go.Surface(x=x_CTR, y=y_CTR, z=z_CTR, colorscale=[[0, 'rgba(0, 0, 0, 0.2)'], [1, 'rgba(0, 0, 0, 0.2)']], hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>', showscale=False, name='CTR Ellipsoid'))
        fig.add_trace(go.Surface(x=x_FR, y=y_FR, z=z_FR, colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(255, 0, 0, 0.2)']], hovertemplate='R_m: %{x}<br>V_thresh: %{y}<br>tau_w_ad: %{z}<br>', showscale=False, name='FR Ellipsoid'))

    # update layout for the 3D plot
    fig.update_layout(
    scene=dict(
        xaxis=dict(
            title=dict(
                text='R<sub>m</sub> / MΩ',  # x-axis label
                font=dict(size=20)), tickfont=dict(size=15), backgroundcolor='white', gridcolor='lightgrey', showbackground=True),
        yaxis=dict(
            title=dict(
                text='V<sub>thresh</sub> / mV',  # y-axis label
                font=dict(size=20)), tickfont=dict(size=15), backgroundcolor='white', gridcolor='lightgrey', showbackground=True),
        zaxis=dict(
            title=dict(
                text='τ<sub>w,ad</sub>',  # z-axis label
                font=dict(size=20)), tickfont=dict(size=15), backgroundcolor='white', gridcolor='lightgrey', showbackground=True),
                aspectmode='cube'),
    #title=title,
    width=850,
    height=850,
    font=dict(
        family='CMU Serif',  # LaTeX-like font
        color='black'),
    legend=dict(
            yanchor='top',
            y=-0.1,
            xanchor='left',
            x=0.05))
    
    fig.show()

    # save the plot as an HTML file
    if savename is not None:
        pio.write_html(fig, '../Figures/Zeldenrust_' + str(savename) + '.html')
    
    return fig



#### PCA plotting ####


def plot_pc_loadings_bars(pca, feature_names, pc='PC1', color='grey', figsize=(3,3), ax=None, savename=None):
    # create horizontal bar plot of loadings for a single PC, sorted by |loading|
    # input
    # pca is a fitted PCA object
    # feature_names is a list of feature names
    # pc is the principal component to plot (e.g. 'PC1', 'PC2')
    # color is the color to plot the bars in
    # figsize sets figure size for standalone figures
    # ax assigns plot to existing axis
    # savename is an optional name to save figure

    # decide if we create a new figure or draw into an existing axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        standalone = True
    else:
        fig = ax.figure
        standalone = False

    # parse PC index from string, e.g. 'PC2' -> 1
    pc_idx = int(pc.replace('PC', '')) - 1

    # get loadings for selected PC
    vals = pca.components_[pc_idx]  # shape (n_features,)

    # sort by absolute loading (ascending, as in your original code)
    order = np.argsort(np.abs(vals))
    vals_sorted = vals[order]
    names_sorted = [feature_names[i] for i in order]

    y_pos = np.arange(len(vals_sorted))

    # color by sign (kept identical to your function)
    colors = [color if v > 0 else color for v in vals_sorted]

    ax.barh(y_pos, vals_sorted, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)

    # labels for y-axis (your translation function)
    clean_labels = [value_key_text_plot_Zeldenrust(name, plot_mode='correlation') for name in names_sorted]
    ax.set_yticklabels(clean_labels)

    ax.set_xlabel("Loading")
    ax.set_title(f"{pc} loadings")

    # vertical line at 0
    ax.axvline(0, color='black', linewidth=1)

    # remove frame (all spines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # save & show only for standalone figures
    if standalone is True:
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()

def plot_3_pc_loadings_bars(pca, feature_names, color='grey', description=False, figsize=(10,4), savename=False):
    # plot horizontal bar plots of PCA loadings for first 3 PCs
    # input
    # pca is a fitted PCA object
    # feature_names is a list of the feature names
    # color is the color to plot the bars in
    # description is a string for the title of the plot
    # figsize determines the figsize
    # savename decides whether the figure is saved or not (False -> don't save)
    
    pcs = ('PC1', 'PC2', 'PC3')

    fig = plt.figure(figsize=figsize)
    grid = gs.GridSpec(1, len(pcs), figure=fig, wspace=0.7)
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(pcs))]

    for pc, ax in zip(pcs, axes):
        # plot loadings into the given axis
        plot_pc_loadings_bars(pca, feature_names, pc=pc, color=color, ax=ax)
    
    if description: 
        fig.suptitle(description, y=1.02)
    
    if savename is not False:
        save_path = f'../Figures/{savename}.pdf'
        fig.savefig(save_path, bbox_inches='tight', transparent=True)
        print(f"saved: {save_path}")

    plt.show()
    
    return fig, axes

def plot_pca_interactive_3D(scores, cell_type, pca, results_exc=None, results_inh=None, feature_value=None, savename=None):
    # 3D scatter plot of the first three principal components with exc/inh colors
    # input
    # scores is an array of length N x M with PCA scores (at least 3 PCs) for each cell
    # cell_type is an array of length N with entries 'exc' or 'inh' for each cell
    # pca is the fitted PCA object containing explained_variance_ratio_
    # results_exc is a dictionary of excitatory cell features
    # results_inh is a dictionary of inhibitory cell features
    # feature_value is a string functioning as a key in results_exc/results_inh to use for color coding         
    # savename is the name to save the plot as html file (without extension)
    # output
    # fig is the 3D interactive Plotly scatter plot of PC1 vs PC2 vs PC3

    # convert to numpy arrays (for boolean indexing)
    scores = np.asarray(scores)
    cell_type = np.asarray(cell_type)

    # split indices
    idx_exc = (cell_type == 'exc')
    idx_inh = (cell_type == 'inh')
    n_exc = idx_exc.sum()
    n_inh = idx_inh.sum()

    # decide whether to use color coding by feature
    if results_exc is not None and results_inh is not None and feature_value is not None:
        use_color = True 
    else: 
        use_color = False
        
    z_exc = z_inh = z_min = z_max = None

    if use_color:
        # extract feature arrays and flatten
        z_exc = np.asarray(results_exc[feature_value]).ravel()
        z_inh = np.asarray(results_inh[feature_value]).ravel()

        # shared normalization across exc & inh, ignoring NaNs
        z_all = np.concatenate([z_exc, z_inh])
        z_min = np.nanmin(z_all)
        z_max = np.nanmax(z_all)

    fig = go.Figure()

    if use_color:
        # exc cells color coded
        fig.add_trace(go.Scatter3d(
            x=scores[idx_exc, 0],
            y=scores[idx_exc, 1],
            z=scores[idx_exc, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=z_exc,                  # color by feature
                colorscale='Viridis',
                cmin=z_min,
                cmax=z_max,
                colorbar=dict(
                    title=feature_value      # you can change to a prettier label if you want
                ),
                line=dict(color='red', width=1.5),
                opacity=0.7),
            name='exc cells',
            hovertemplate=(
                'PC1: %{x:.3f}<br>'
                'PC2: %{y:.3f}<br>'
                'PC3: %{z:.3f}<br>'
                f"{feature_value}: %{{marker.color:.3f}}<br>"
                'Type: exc<extra></extra>'
            )))
    else:
        # excitatory cells (red)
        fig.add_trace(go.Scatter3d(
            x=scores[idx_exc, 0],
            y=scores[idx_exc, 1],
            z=scores[idx_exc, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='red',
                line=dict(color='red', width=1.5),
                opacity=0.7),
            name='exc cells',
            hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<br>Type: exc<extra></extra>'))

    if use_color:
        # inh cells color coded
        fig.add_trace(go.Scatter3d(
            x=scores[idx_inh, 0],
            y=scores[idx_inh, 1],
            z=scores[idx_inh, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=z_inh,                  # color by feature
                colorscale='Viridis',
                cmin=z_min,
                cmax=z_max,
                showscale=False,              # use shared colorbar from exc trace
                line=dict(color='blue', width=1.5),
                opacity=0.7),
            name='inh cells',
            hovertemplate=(
                'PC1: %{x:.3f}<br>'
                'PC2: %{y:.3f}<br>'
                'PC3: %{z:.3f}<br>'
                f"{feature_value}: %{{marker.color:.3f}}<br>"
                'Type: inh<extra></extra>'
            )))
    else:
        # inhibitory cells (blue)
        fig.add_trace(go.Scatter3d(
            x=scores[idx_inh, 0],
            y=scores[idx_inh, 1],
            z=scores[idx_inh, 2],
            mode='markers',
            marker=dict(
                size=6,
                color='blue',
                line=dict(color='blue', width=1.5),
                opacity=0.7),
            name='inh cells',
            hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<br>Type: inh<extra></extra>'))

    # axis labels with explained variance
    evr = pca.explained_variance_ratio_
    x_label = f"PC1 ({evr[0]*100:.1f}% var.)"
    y_label = f"PC2 ({evr[1]*100:.1f}% var.)"
    z_label = f"PC3 ({evr[2]*100:.1f}% var.)"

    # layout in the same style as your Zeldenrust 3D function
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(
                    text=x_label,
                    font=dict(size=20)),
                tickfont=dict(size=15),
                backgroundcolor='white',
                gridcolor='lightgrey',
                showbackground=True),
            yaxis=dict(
                title=dict(
                    text=y_label,
                    font=dict(size=20)),
                tickfont=dict(size=15),
                backgroundcolor='white',
                gridcolor='lightgrey',
                showbackground=True),
            zaxis=dict(
                title=dict(
                    text=z_label,
                    font=dict(size=20)),
                tickfont=dict(size=15),
                backgroundcolor='white',
                gridcolor='lightgrey',
                showbackground=True),
            aspectmode='cube'),
        width=850,
        height=850,
        font=dict(
            family='CMU Serif',
            color='black'),
        legend=dict(
            yanchor='top',
            y=-0.1,
            xanchor='left',
            x=0.05))

    fig.show()

    # save the plot as an HTML file
    if savename is not None:
        pio.write_html(fig, '../Figures/' + str(savename) + '_PC1_PC2_PC3.html')

    return fig

######### plot PCA & grid data ##########


def translate_axes_grid_to_exp(grid_axes):
    # translate grid_axes (grid naming) to exp axes naming
    # input
    # grid_axes is a tuple/list of 3 axis keys in grid naming
    
    # output
    # exp_axes is a tuple of 3 axis keys in exp naming (or None if unknown)

    GRID_TO_EXP_AXIS_KEY = {
        'pc1': 'pc1',
        'pc2': 'pc2',
        'pc3': 'pc3',
        'E_L': 'E_L_mV_list',
        'V_thresh': 'V_thresh_mV_list',
        'R_m': 'R_m_mean_MOhm_list',
        'C_m': 'C_m_mean_pF_list',
        'Delta_T_ad': 'Delta_T_mean_mV_list',
        'V_reset': 'V_reset_mean_mV_list',
        'tau_w_ad': 'tau_w_mean_ms_list',
        'a_ad': 'a_w_mean_nS_list',
        'b_ad': 'b_w_mean_nA_list'}

    # opposite mapping (EXP -> GRID)
    #EXP_TO_GRID_AXIS_KEY = {v: k for k, v in GRID_TO_EXP_AXIS_KEY.items()}

    exp_axes = []
    for ax in grid_axes:
        exp_axes.append(GRID_TO_EXP_AXIS_KEY.get(ax, None))
    return tuple(exp_axes)


def translate_value_key_grid_to_exp(value_key):
    # translate grid value_key (grid naming) to exp value_key naming
    # input
    # value_key is a grid value key (or None)
    
    # output
    # exp_value_key is the corresponding exp key (or None)

    # translation dictionaries (GRID reference -> EXP keys)
    
    GRID_TO_EXP_VALUE_KEY = {
    'r_post': 'firing_rate_Hz_list',
    'E_tot': 'E_tot_1e9_ATP_per_s_list',
    'MI': 'MI_FZ_bits_list',
    'MI_per_energy': 'MI_FZ_per_energy_list',
    'MICE': 'MICE_FZ_list',
    'MICE_per_energy': 'MICE_FZ_per_energy_list',
    
    # no exp equivalent by default:
    'hit_fraction': None,
    'false_alarm_fraction': None,
    
    # grid-only metrics:
    'CV_V_m': None,
    'CV_ISI': None,
    'CV_ISI_per_energy': None}
    
    if value_key is None:
        return None
    return GRID_TO_EXP_VALUE_KEY.get(value_key, None)


def extract_xyz_from_grid_dict(grid_data, axes=('pc1','pc2','pc3')):
    # extract x,y,z arrays from grid dict-of-dicts for chosen axes
    # input
    # grid_data is dict-of-dicts (e.g. results from Zeldenrust_single_grid_run_PCA)
    # axes is a tuple/list of 3 keys in the grid entry dict

    # output
    # x, y, z are arrays of length n_points
    # keys is the list of dict keys in the same order

    import numpy as np

    ax1, ax2, ax3 = axes

    x, y, z = [], [], []
    keys = []

    for k, d in grid_data.items():
        if (ax1 not in d) or (ax2 not in d) or (ax3 not in d):
            raise KeyError(f"grid_axes key missing for grid entry '{k}'. Needed: {axes}")

        x.append(float(d[ax1]))
        y.append(float(d[ax2]))
        z.append(float(d[ax3]))
        keys.append(k)

    return np.asarray(x), np.asarray(y), np.asarray(z), keys


def extract_xyz_from_exp_dict(exp_data, exp_axes=('pc1','pc2','pc3')):
    # extract x,y,z arrays from EXP dict-of-dicts (build_dic_exp_data output)
    # input
    # exp_data is dict-of-dicts from build_dic_exp_data
    # exp_axes is a tuple/list of 3 keys in exp naming
    #   - if exp_axes are ('pc1','pc2','pc3'), coordinates are taken from entry['scores'][0:3]
    #   - otherwise coordinates are taken from entry[exp_axes[i]] (must exist in entry dict)
    
    # output
    # x, y, z are arrays of length n_points
    # keys is the list of dict keys in the same order

    import numpy as np

    ax1, ax2, ax3 = exp_axes

    x, y, z = [], [], []
    keys = []

    for k, d in exp_data.items():
        if exp_axes == ('pc1','pc2','pc3'):
            if 'scores' not in d:
                raise KeyError(f"Experimental entry '{k}' has no 'scores' key (needed for PC axes).")
            sc = np.asarray(d['scores']).ravel()
            if len(sc) < 3:
                raise ValueError(f"Experimental entry '{k}' has scores of length {len(sc)} (<3).")
            x.append(float(sc[0]))
            y.append(float(sc[1]))
            z.append(float(sc[2]))
        else:
            if (ax1 not in d) or (ax2 not in d) or (ax3 not in d):
                raise KeyError(f"exp_axes key missing for exp entry '{k}'. Needed: {exp_axes}")
            x.append(float(d[ax1]))
            y.append(float(d[ax2]))
            z.append(float(d[ax3]))

        keys.append(k)

    return np.asarray(x), np.asarray(y), np.asarray(z), keys

def get_value_from_entry(entry_dict, value_key, mi_index=7):
    # get value for color coding from one results entry
    #
    # input
    # entry_dict is one dictionary entry
    # value_key is the key to extract
    # mi_index is the MI_vec index to use if value_key == 'MI_vec'
    
    # output
    # value (float or nan)

    import numpy as np

    v = entry_dict.get(value_key, np.nan)

    if value_key == 'MI_vec' and v is not None and hasattr(v, '__len__'):
        try:
            v = v[mi_index]
        except Exception:
            v = np.nan

    if v is None:
        v = np.nan

    return float(v) if np.isfinite(v) else np.nan

def make_hovertemplate(axis_labels, value_label=None, extra_lines=None):
    # create a consistent hovertemplate
    # input
    # axis_labels is a tuple (xlab, ylab, zlab)
    # value_label is optional label for marker color
    # extra_lines is optional list of additional lines (strings), e.g. ["Type: exc"]
    
    # output
    # hovertemplate string

    xlab, ylab, zlab = axis_labels

    lines = [
        f"{xlab}: %{{x:.3f}}",
        f"{ylab}: %{{y:.3f}}",
        f"{zlab}: %{{z:.3f}}"]

    if value_label is not None:
        lines.append(f"{value_label}: %{{marker.color:.3f}}")

    if extra_lines is not None:
        lines.extend(extra_lines)

    return "<br>".join(lines) + "<extra></extra>"



def plot_grid_and_exp_interactive_3D(grid_data_exc=None, grid_data_inh=None, exp_data_exc=None, exp_data_inh=None, value_key=None, normalization_mode="shared_axis", grid_axes=('pc1','pc2','pc3'), interpolation=False, lower_threshold=None, upper_threshold=None, color_exc='coral', color_inh='darkcyan', grid_marker_size=4, exp_marker_size=6, title=None, colorbar_mode=True, legend_mode=False, axes_mode=False, base_font_size=40, eye=dict(x=-1.8, y=1.6, z=0.8), show_mode=True, savename=None):
    # 3D interactive Plotly plot for experimental and simulated grid data (exc/inh), with name translation
    #
    # input
    # grid_data_exc is a results dict (dict-of-dicts) for the excitatory simulated grid (or None)
    # grid_data_inh is a results dict (dict-of-dicts) for the inhibitory simulated grid (or None)
    # exp_data_exc is a results dict (dict-of-dicts) for the excitatory experimental data (or None)
    # exp_data_inh is a results dict (dict-of-dicts) for the inhibitory experimental data (or None)
    # value_key is the GRID reference key
    # normalization_mode decides whether to use a "shared_axis" or "normalized_axis"
    # grid_axes is a tuple of 3 GRID reference axis keys (translated for EXP)
    # interpolation determines if the actual simulated data points should be exchanged by a contionous interpolated plane
    # lower_threshold, upper_threshold is the value under or over which the points should be transparent
    # color_exc, color_inh are the excitatory/inhibitory colors if default coloring is used
    # grid_marker_size, exp_marker_size set marker sizes
    # title is plot title
    # colorbar_mode defines if the color bar should be plotted or not 
    # legend_mode defines if the legend should be plotted or not 
    # axes_mode defines if the major axes are plotted with thick lines 
    # base_font_size is the base font size 
    # eye is the dictionary of the initial camera position 
    # show_mode determines if the fig gets displayed or not 
    # savename is the name to save the grid as html file
    
    # output
    # fig is the 3D interactive Plotly figure

    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio
    
        
    fig = go.Figure()
    
    def add_interpolated_volume_grid(data, celltype, base_color, resolution=30):
        # interpolate grid values in (x,y,z) space and plot as go.Volume, only used for GRID data when interpolation=True.
        
        if data is None or value_key is None:
            return

        # extract grid xyz + keys
        x, y, z, keys = extract_xyz_from_grid_dict(data, axes=grid_axes)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        vals_raw = np.asarray([get_value_from_entry(data[k], value_key) for k in keys], dtype=float)

        # valid values only (keep your 0 invalid convention)
        valid = threshold_mask(vals_raw)
        if not np.any(valid):
            return

        # choose value mapping + shared axis bounds (consistent with your markers)
        if normalization_mode == "normalized_axis":
            vals_color = normalize_vals(vals_raw, grid_min, grid_max)
            cmin, cmax = 0.0, 1.0
        else:
            vals_color = vals_raw
            cmin, cmax = z_min, z_max

        # build interpolation grid
        xi = np.linspace(float(np.min(x[valid])), float(np.max(x[valid])), resolution)
        yi = np.linspace(float(np.min(y[valid])), float(np.max(y[valid])), resolution)
        zi = np.linspace(float(np.min(z[valid])), float(np.max(z[valid])), resolution)
        X, Y, Z = np.meshgrid(xi, yi, zi, indexing="xy")

        # interpolate
        V = griddata(
            points=np.column_stack([x[valid], y[valid], z[valid]]),
            values=vals_color[valid],
            xi=(X, Y, Z),
            method="linear"
        )
        # apply thresholds to interpolated values
        if lower_threshold is not None:
            V[V < lower_threshold] = np.nan
        if upper_threshold is not None:
            V[V > upper_threshold] = np.nan

        # if everything is NaN after interpolation, skip
        if V is None or np.all(~np.isfinite(V)):
            return

        # plot as volume (continuous look)
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=np.nan_to_num(V, nan=np.nan).flatten(),
            # keep your shared axis
            isomin=(lower_threshold if lower_threshold is not None else cmin),
            isomax=(upper_threshold if upper_threshold is not None else cmax),
            opacity=0.08,           # tweak as you like
            surface_count=6,        # tweak as you like
            colorscale="Viridis",
            showscale=False,        # your shared dummy colorbar handles this
            hovertemplate=make_hovertemplate(
                axis_labels_hover,
                value_label=(grid_value_label if normalization_mode == "shared_axis" else f"{grid_value_label} (norm)"),
                extra_lines=[f"Type: {celltype}", "Data: grid (interp)"]
            ),
            name=f"{celltype} grid (interp)",
            showlegend=False
        ))
    
    # translate exp axis keys / exp value_key
    exp_axes = translate_axes_grid_to_exp(grid_axes)
    exp_value_key = translate_value_key_grid_to_exp(value_key)

    # if translation fails for any axis: do not plot exp in that space
    plot_exp = (None not in exp_axes)

    # rule 1: grid-only value_key -> exp stays default
    if value_key in ('CV_V_m', 'CV_ISI', 'CV_ISI_per_energy'):
        exp_value_key = None

    # rule 2: exp-only value_key is not requested here (your value_key is GRID reference)
    # if someone passes an exp-only key by accident -> grid stays default
    if value_key in ('MI_calculated_bits_list', 'MI_calculated_per_energy_list', 'FI_calculated_list', 'FI_FZ_list'):
        value_key = None
        exp_value_key = None

    
    # shared color normalization (only if we are value-coloring something)
    z_min, z_max = None, None         # used in shared_axis mode (raw values)
    grid_min, grid_max = None, None   # used in normalized_axis mode (grid-only)
    exp_min, exp_max = None, None     # used in normalized_axis mode (exp-only)

    grid_color_mode = (value_key is not None)
    exp_color_mode  = (exp_value_key is not None)
    
    def collect_valid_vals(results_dict, key):
        vals = []
        if results_dict is None or key is None:
            return vals
        for k in results_dict.keys():
            v = get_value_from_entry(results_dict[k], key)
            if np.isfinite(v) and (v != 0):  # keep your "0 invalid/empty" convention
                vals.append(float(v))
        return vals

    grid_vals = []
    exp_vals  = []

    if grid_color_mode:
        grid_vals += collect_valid_vals(grid_data_exc, value_key)
        grid_vals += collect_valid_vals(grid_data_inh, value_key)

    if exp_color_mode and plot_exp:
        exp_vals += collect_valid_vals(exp_data_exc, exp_value_key)
        exp_vals += collect_valid_vals(exp_data_inh, exp_value_key)

    # bounds for normalized_axis
    if len(grid_vals) > 0:
        grid_min, grid_max = float(np.min(grid_vals)), float(np.max(grid_vals))
    if len(exp_vals) > 0:
        exp_min, exp_max = float(np.min(exp_vals)), float(np.max(exp_vals))

    # bounds for shared_axis (raw shared colorbar)
    all_vals = []
    all_vals += grid_vals
    all_vals += exp_vals
    if len(all_vals) > 0:
        z_min, z_max = float(np.min(all_vals)), float(np.max(all_vals))

    def normalize_vals(vals, vmin, vmax):
        # returns array in [0,1]; safe for constant ranges
        denom = (vmax - vmin) if (vmin is not None and vmax is not None) else None
        if denom is None or denom == 0:
            # if all values identical, map them to 0.5 (arbitrary but stable)
            return np.full_like(vals, 0.5, dtype=float)
        return (vals - vmin) / denom
    
    def threshold_mask(vals):
        
        # returns boolean mask of values within thresholds
        
        mask = np.isfinite(vals) & (vals != 0)
        if lower_threshold is not None:
            mask &= (vals >= lower_threshold)
        if upper_threshold is not None:
            mask &= (vals <= upper_threshold)
        return mask
    
    def split_threshold_masks(vals):
        # valid: finite & non-zero (your convention)
        # inside: valid and within thresholds
        # outside: valid but outside thresholds
        
        valid = np.isfinite(vals) & (vals != 0)
    
        inside = valid.copy()
        if lower_threshold is not None:
            inside &= (vals >= lower_threshold)
        if upper_threshold is not None:
            inside &= (vals <= upper_threshold)
    
        outside = valid & (~inside)
        invalid = ~valid
        
        return inside, outside, invalid
    
    outside_opacity_factor = 0.2  # outside-threshold points are 20% as opaque

    axis_labels_axis = tuple(value_key_text_plot_Zeldenrust(ax, plot_mode="grid_axis")  for ax in grid_axes)
    axis_labels_hover = tuple(value_key_text_plot_Zeldenrust(ax, plot_mode="grid_hover") for ax in grid_axes)
    
    grid_value_label = value_key_text_plot_Zeldenrust(value_key, plot_mode='grid_hover') if grid_color_mode else None
    if normalization_mode=="normalized_axis":
        grid_value_label = "MI (norm.)"
    exp_value_label  = value_key_text_plot_Zeldenrust(value_key, plot_mode='grid_hover') if exp_color_mode else None

    # helper: colorbar spec like in your other function
    scaling_factor_colorbar_label = 0.8
    scaling_factor_colorbar_numbers = 0.75
    scaling_factor_ticks_numbers = 0.5
    scaling_factor_ticks_labels = 0.9

    if base_font_size > 49:
        scaling_factor_colorbar_label = 0.9
        scaling_factor_colorbar_numbers = 1.0
        scaling_factor_ticks_numbers = 0.45
        scaling_factor_ticks_labels = 0.9

    def build_colorbar(title_text):
        if isinstance(colorbar_mode, dict):
            return colorbar_mode
        return dict(
            title=dict(
                text=title_text,
                font=dict(size=base_font_size * scaling_factor_colorbar_label, color='black')),
            tickmode="auto",
            nticks=3,
            tickfont=dict(size=base_font_size * scaling_factor_colorbar_numbers, color='black'),
            tickcolor='black',
            outlinecolor='black',
            outlinewidth=2,
            len=0.75,          # match your other function's global enforcement
            thickness=20,      # match
            x=0.95,
            y=0.425,
            yanchor="middle")
    
    def add_trace(data, is_exp, kind, celltype, base_color):
        if data is None:
            return

        if is_exp:
            if not plot_exp:
                return
            x, y, z, keys = extract_xyz_from_exp_dict(data, exp_axes=exp_axes)
            vk = exp_value_key
            vlabel = exp_value_label
        else:
            x, y, z, keys = extract_xyz_from_grid_dict(data, axes=grid_axes)
            vk = value_key
            vlabel = grid_value_label

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        if vk is None:
            # default type coloring
            opacity = 0.75 if kind == 'exp' else 0.4
            size = exp_marker_size if kind == 'exp' else grid_marker_size

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=size, color=base_color, opacity=opacity),
                name=f'{celltype} {kind}',
                showlegend=bool(legend_mode),
                hovertemplate=make_hovertemplate(
                    axis_labels_hover,
                    value_label=None,
                    extra_lines=[f"Type: {celltype}", f"Data: {kind}"]
                )
            ))
            return

        # value coloring + outline indicates type
        vals_raw = np.asarray([get_value_from_entry(data[k], vk) for k in keys], dtype=float)

        size = exp_marker_size if kind == 'exp' else grid_marker_size
        base_opacity = 0.7 if kind == 'exp' else 0.45
        
        # match your convention: treat 0 as "invalid/empty"
        inside_mask, outside_mask, invalid_mask = split_threshold_masks(vals_raw)
        
        if normalization_mode == "normalized_axis":
            if is_exp:
                vals_color = normalize_vals(vals_raw, exp_min, exp_max)
            else:
                vals_color = normalize_vals(vals_raw, grid_min, grid_max)
            cmin, cmax = 0.0, 1.0
        else:
            vals_color = vals_raw
            cmin, cmax = z_min, z_max
        
        # if bounds are missing, let plotly autoscale
        cauto_flag = not (cmin is not None and cmax is not None)

        

        # ---- valid values ----
        if np.any(inside_mask):
            fig.add_trace(go.Scatter3d(
                x=x[inside_mask], y=y[inside_mask], z=z[inside_mask],
                mode='markers',
                marker=dict(
                    size=size,
                    color=vals_color[inside_mask],
                    cmin=cmin, cmax=cmax,
                    colorscale='Viridis',
                    cauto=cauto_flag,
                    showscale=False,
                    opacity=base_opacity,
                    line=dict(color=base_color, width=1.5)
                ),
                name=f'{celltype} {kind}',
                showlegend=bool(legend_mode),
                hovertemplate=make_hovertemplate(
                    axis_labels_hover,
                    value_label=vlabel,
                    extra_lines=[f"Type: {celltype}", f"Data: {kind}"]
                )
            ))
        if np.any(outside_mask):
            fig.add_trace(go.Scatter3d(
                x=x[outside_mask], y=y[outside_mask], z=z[outside_mask],
                mode='markers',
                marker=dict(
                    size=size,
                    color=vals_color[outside_mask],
                    cmin=cmin, cmax=cmax,
                    colorscale='Viridis',
                    cauto=cauto_flag,
                    showscale=False,
                    opacity=base_opacity * outside_opacity_factor,
                    line=dict(color=base_color, width=1.0)
                ),
                name=f'{celltype} {kind} (out)',
                showlegend=False,
                hovertemplate=make_hovertemplate(
                    axis_labels_hover,
                    value_label=vlabel,
                    extra_lines=[f"Type: {celltype}", f"Data: {kind}", "Outside threshold"]
                )
            ))

        # ---- invalid values ----
        if np.any(invalid_mask):
            fig.add_trace(go.Scatter3d(
                x=x[invalid_mask], y=y[invalid_mask], z=z[invalid_mask],
                mode='markers',
                marker=dict(
                    size=size,
                    color='rgba(0,0,0,0)',
                    line=dict(color=base_color, width=1.5),
                    opacity=1.0
                ),
                name=f'{celltype} {kind} 0',
                showlegend=False,
                hovertemplate=make_hovertemplate(
                    axis_labels_hover,
                    value_label=None,
                    extra_lines=[f"Type: {celltype}", f"Data: {kind}", f"{vlabel}: 0"]
                )
            ))

    # add traces (same order as before)
    add_trace(exp_data_exc, is_exp=True,  kind='exp',  celltype='exc', base_color=color_exc)
    add_trace(exp_data_inh, is_exp=True,  kind='exp',  celltype='inh', base_color=color_inh)
    if interpolation:
        # interpolated grid volumes
        add_interpolated_volume_grid(grid_data_exc, celltype="exc", base_color=color_exc)
        add_interpolated_volume_grid(grid_data_inh, celltype="inh", base_color=color_inh)
    else:
        # discrete grid dots
        add_trace(grid_data_exc, is_exp=False, kind='grid', celltype='exc', base_color=color_exc)
        add_trace(grid_data_inh, is_exp=False, kind='grid', celltype='inh', base_color=color_inh)


    # shared colorbar (dummy trace) if enabled
    if (grid_color_mode or exp_color_mode) and bool(colorbar_mode):
        if normalization_mode == "normalized_axis":
            cb_min, cb_max = 0.0, 1.0
        else:
            cb_min, cb_max = z_min, z_max

        if (cb_min is not None) and (cb_max is not None):
            colorbar_title = grid_value_label if grid_value_label is not None else exp_value_label

            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(
                    size=1,
                    color=[cb_min, cb_max],
                    colorscale='Viridis',
                    cmin=cb_min,
                    cmax=cb_max,
                    cauto=not (cb_min is not None and cb_max is not None),
                    colorbar=build_colorbar(colorbar_title),
                    showscale=True
                ),
                showlegend=False,
                hoverinfo='skip'
            ))

    # layout: fonts / camera / legend / axes
    if title is None:
        title = f"{grid_axes[0]} vs {grid_axes[1]} vs {grid_axes[2]}"

    # tick sizing
    #axis_title_size = base_font_size * scaling_factor_ticks_labels
    #ätick_size = base_font_size * scaling_factor_ticks_numbers

    # Explicit axis lines (like your other function) if axes_mode
    if axes_mode:
        # estimate bounds from all plotted data (use traces already added)
        xs, ys, zs = [], [], []
        for tr in fig.data:
            if isinstance(tr, go.Scatter3d) and tr.x is not None:
                xv = np.asarray([v for v in tr.x if v is not None], dtype=float) if tr.x is not None else np.array([])
                yv = np.asarray([v for v in tr.y if v is not None], dtype=float) if tr.y is not None else np.array([])
                zv = np.asarray([v for v in tr.z if v is not None], dtype=float) if tr.z is not None else np.array([])
                if xv.size: xs.append(np.nanmin(xv)); xs.append(np.nanmax(xv))
                if yv.size: ys.append(np.nanmin(yv)); ys.append(np.nanmax(yv))
                if zv.size: zs.append(np.nanmin(zv)); zs.append(np.nanmax(zv))

        if xs and ys and zs:
            x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
            y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
            z_min_ax, z_max_ax = float(np.nanmin(zs)), float(np.nanmax(zs))

            def edge_ticks(vmin, vmax):
                return [vmin + 0.1*(vmax-vmin), vmin + 0.9*(vmax-vmin)]

            x0, x1 = edge_ticks(x_min, x_max)
            y0, y1 = edge_ticks(y_min, y_max)
            z0, z1 = edge_ticks(z_min_ax, z_max_ax)

            axis_line_width = 8

            # origin corner
            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y0], z=[z0, z0],
                mode='lines',
                line=dict(width=axis_line_width, color='black'),
                hoverinfo='skip',
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[x0, x0], y=[y0, y1], z=[z0, z0],
                mode='lines',
                line=dict(width=axis_line_width, color='black'),
                hoverinfo='skip',
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[x0, x0], y=[y0, y0], z=[z0, z1],
                mode='lines',
                line=dict(width=axis_line_width, color='black'),
                hoverinfo='skip',
                showlegend=False
            ))
    
    def format_tick(v):
        return f"{int(round(v))}" if abs(v) > 2 else f"{v:.2f}"

    x_all, y_all, z_all = [], [], []

    for tr in fig.data:
        # only use real point traces (skip dummy colorbar trace with None)
        if isinstance(tr, go.Scatter3d) and tr.x is not None: #if hasattr(tr, "x") and tr.x is not None:
            x_all.extend([v for v in tr.x if v is not None])
        if isinstance(tr, go.Scatter3d) and tr.y is not None: #if hasattr(tr, "y") and tr.y is not None:
            y_all.extend([v for v in tr.y if v is not None])
        if isinstance(tr, go.Scatter3d) and tr.z is not None: #if hasattr(tr, "z") and tr.z is not None:
            z_all.extend([v for v in tr.z if v is not None])
    
    # get min/max of axis
    x_min, x_max = float(min(x_all)), float(max(x_all))
    y_min, y_max = float(min(y_all)), float(max(y_all))
    z_min, z_max = float(min(z_all)), float(max(z_all))
    
    # position ticks at 10% & 90% of the length
    def edge_ticks(vmin, vmax):
        lower = vmin + 0.1 * (vmax - vmin)
        upper = vmin + 0.9 * (vmax - vmin)
        return [lower, upper]
    
    x_ticks = edge_ticks(x_min, x_max)
    y_ticks = edge_ticks(y_min, y_max)
    z_ticks = edge_ticks(z_min, z_max)
    
    # thicker axis line style if enabled by axes_mode
    axis_line_width = 8 if axes_mode else 2
    

    fig.update_layout(
        font=dict(family='Helvetica', color='black'),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text=axis_labels_axis[0],
                    font=dict(size=base_font_size * scaling_factor_ticks_labels, color='black')
                ),
                tickmode='array',
                tickvals=x_ticks,
                ticktext=[format_tick(x_ticks[0]), format_tick(x_ticks[1])],
                tickfont=dict(size=base_font_size * scaling_factor_ticks_numbers, color='black'),
                showbackground=True,
                backgroundcolor='white',
                gridcolor='lightgrey'
            ),
            yaxis=dict(
                title=dict(
                    text=axis_labels_axis[1],
                    font=dict(size=base_font_size * scaling_factor_ticks_labels, color='black')
                ),
                tickmode='array',
                tickvals=y_ticks,
                ticktext=[format_tick(y_ticks[0]), format_tick(y_ticks[1])],
                tickfont=dict(size=base_font_size * scaling_factor_ticks_numbers, color='black'),
                showbackground=True,
                backgroundcolor='white',
                gridcolor='lightgrey'
            ),
            zaxis=dict(
                title=dict(
                    text=axis_labels_axis[2],
                    font=dict(size=base_font_size * scaling_factor_ticks_labels, color='black')
                ),
                tickmode='array',
                tickvals=z_ticks,
                ticktext=[format_tick(z_ticks[0]), format_tick(z_ticks[1])],
                tickfont=dict(size=base_font_size * scaling_factor_ticks_numbers, color='black'),
                showbackground=True,
                backgroundcolor='white',
                gridcolor='lightgrey'
            ),
            aspectmode='cube'
        ),
        scene_camera=dict(eye=eye),
        #title=title,
        width=850,
        height=850,
        margin=dict(l=0, r=0, t=0, b=0))

    if show_mode:
        fig.show()

    if savename is not None:
        pio.write_html(fig, '../Figures/' + str(savename) + '.html')

    return fig


# 3D matplotlib Zeldenrust plotting 

def plot_grid_and_exp_3D_matplotlib(grid_data_exc=None, grid_data_inh=None, results_analysis_PCA_analyzed_exc=None, results_analysis_PCA_analyzed_inh=None, exp_data_exc=None, exp_data_inh=None, value_key=None, normalization_mode="shared_axis", grid_axes=('pc1', 'pc2', 'pc3'), interpolation=False, lower_threshold=None, upper_threshold=None, color_exc='coral', color_inh='darkcyan', grid_marker_size=4, exp_marker_size=6, title=None, colorbar_mode=True, legend_mode=False, axes_mode=False, base_font_size=40, elev=20, azim=120, figsize=(4,4), show_mode=True, ax=None, savename=None, dpi=250, cmap='viridis'):
    # plot 3D visualization of grid and experimental data in the same space
    # input
    # grid_data_exc is a results dict (dict-of-dicts) for the excitatory simulated grid (or None)
    # grid_data_inh is a results dict (dict-of-dicts) for the inhibitory simulated grid (or None)
    # results_analysis_PCA_analyzed_exc is a results dict (dict-of-dicts) for the excitatory experimental data projected into PCA space
    # results_analysis_PCA_analyzed_inh is a results dict (dict-of-dicts) for the inhibitory experimental data projected into PCA space
    # exp_data_exc is a results dict (dict-of-dicts) for the excitatory experimental data (or None)
    # exp_data_inh is a results dict (dict-of-dicts) for the inhibitory experimental data (or None)
    # value_key is the grid reference key used for value coloring
    # normalization_mode decides whether to use a "shared_axis" or "normalized_axis" color mapping
    # grid_axes is a tuple of 3 grid reference axis keys, translated for experimental axes
    # interpolation is False or True and enables plotly-like volume via stacked isosurfaces
    # lower_threshold is the lower threshold for filtering and interpolation range, defaults to global min if None
    # upper_threshold is the upper threshold for filtering and interpolation range, defaults to global max if None
    # color_exc is the outline or default color for excitatory data
    # color_inh is the outline or default color for inhibitory data
    # grid_marker_size is marker size for grid points
    # exp_marker_size is marker size for experimental points
    # title is plot title
    # colorbar_mode defines if the color bar should be plotted or not
    # legend_mode defines if the legend should be plotted or not
    # axes_mode defines if the major axes are plotted with thick lines
    # base_font_size is the base font size
    # elev and azim control the camera angle
    # show_mode determines if the figure gets displayed or not
    # figsize sets the figure size for standalone plotting
    # ax assigns plot to an existing axes, if None a new figure and axes are created
    # savename is the name to save the figure as pdf and png
    # dpi is the save resolution
    # cmap is the colormap used
    # output
    # fig is the matplotlib figure
    # ax is the matplotlib 3d axes

    standalone = ax is None
    fig = None
    if standalone:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
        

    def collect_valid_vals(results_dict, key):
        # collect finite non-zero values from a dict-of-dicts for a given key
        # input
        # results_dict is a dict of entries
        # key is the scalar key to extract from each entry
        # output
        # vals is a list of finite non-zero scalar values
        vals = []
        if results_dict is None or key is None:
            return vals
        for k in results_dict.keys():
            v = get_value_from_entry(results_dict[k], key)
            if np.isfinite(v) and (v != 0):
                vals.append(float(v))
        return vals
    
    def normalize_vals(vals, vmin, vmax):
        # normalize array into [0,1] with stable behavior for constant range
        # input
        # vals is a numpy array of values
        # vmin is minimum value
        # vmax is maximum value
        # output
        # out is vals mapped into [0,1]
        denom = (vmax - vmin) if (vmin is not None and vmax is not None) else None
        if denom is None or denom == 0:
            return np.full_like(vals, 0.5, dtype=float)
        return (vals - vmin) / denom
    
    def split_threshold_masks(vals, lower_threshold, upper_threshold):
        # split values into inside, outside, and invalid masks using your 0 invalid convention
        # input
        # vals is a numpy array of scalar values
        # lower_threshold is the minimum value considered inside
        # upper_threshold is the maximum value considered inside
        # output
        # inside_mask is True for finite non-zero values inside thresholds
        # outside_mask is True for finite non-zero values outside thresholds
        # invalid_mask is True for values that are NaN or 0
        valid = np.isfinite(vals) & (vals != 0)
        inside = valid.copy()
        inside &= (vals >= lower_threshold) & (vals <= upper_threshold)
        outside = valid & (~inside)
        invalid = ~valid
        return inside, outside, invalid
    
    def get_xyz_and_vals(data, is_exp):
        # extract x,y,z arrays and scalar values for either grid or exp data
        # input
        # data is a dict-of-dicts
        # is_exp selects extraction from experimental dict space if True
        # output
        # x,y,z are numpy arrays of coordinates
        # vals_raw is numpy array of raw scalar values or None
        if data is None:
            return None, None, None, None
    
        if is_exp:
            if not plot_exp:
                return None, None, None, None
            x, y, z, keys = extract_xyz_from_exp_dict(data, exp_axes=exp_axes)
            vk = exp_value_key
        else:
            x, y, z, keys = extract_xyz_from_grid_dict(data, axes=grid_axes)
            vk = value_key
    
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)
    
        if vk is None:
            vals_raw = None
        else:
            vals_raw = np.asarray([get_value_from_entry(data[k], vk) for k in keys], dtype=float)
    
        return x, y, z, vals_raw
    
    def add_scatter(x, y, z, vals_raw, kind, celltype, base_color, size, base_opacity):
        # add scatter points with the same threshold logic as your grid plot
        # input
        # x,y,z are numpy arrays of coordinates
        # vals_raw is numpy array of raw scalar values or None
        # kind is 'grid' or 'exp'
        # celltype is 'exc' or 'inh'
        # base_color is the outline or fill color for type identity
        # size is marker size
        # base_opacity is marker opacity
        # output
        # returns nothing
        if x is None:
            return
    
        if vals_raw is None:
            ax.scatter(x, y, z, s=size, c=base_color, alpha=base_opacity, depthshade=False, label=f"{celltype} {kind}" if legend_mode else None)
            return
    
        inside_mask, outside_mask, invalid_mask = split_threshold_masks(vals_raw, lower_threshold, upper_threshold)
    
        if normalization_mode == "normalized_axis":
            if kind == "exp":
                vals_color = normalize_vals(vals_raw, float(np.min(exp_vals)) if len(exp_vals) else 0.0, float(np.max(exp_vals)) if len(exp_vals) else 1.0)
            else:
                vals_color = normalize_vals(vals_raw, float(np.min(grid_vals)) if len(grid_vals) else 0.0, float(np.max(grid_vals)) if len(grid_vals) else 1.0)
        else:
            vals_color = vals_raw
    
        # inside threshold points
        if np.any(inside_mask):
            ax.scatter(x[inside_mask], y[inside_mask], z[inside_mask], s=size, c=vals_color[inside_mask], cmap=cmap, norm=norm, alpha=base_opacity, depthshade=False, edgecolors=base_color, linewidths=0.3, label=f"{celltype} {kind}" if legend_mode else None)
    
        # outside threshold points
        if np.any(outside_mask):
            ax.scatter(x[outside_mask], y[outside_mask], z[outside_mask], s=size, c=vals_color[outside_mask], cmap=cmap, norm=norm, alpha=base_opacity * outside_opacity_factor, depthshade=False, edgecolors=base_color, linewidths=0.3)
    
        # invalid points as empty markers
        if np.any(invalid_mask):
            ax.scatter(x[invalid_mask], y[invalid_mask], z[invalid_mask], s=size, facecolors='none', edgecolors=base_color, linewidths=0.3, alpha=1.0, depthshade=False)
    
    def plot_isosurface_from_volume(ax, vol, x_lin, y_lin, z_lin, level, norm, cmap='viridis', alpha=0.25):
        # robust marching-cubes wrapper: clamps/guards level against actual vol range
        vmin = float(np.min(vol))
        vmax = float(np.max(vol))
    
        # marching_cubes is strict about range
        if not (vmin <= float(level) <= vmax):
            return None
    
        try:
            verts_ijk, faces, _, _ = marching_cubes(vol, level=float(level))
        except ValueError:
            return None
    
        nx, ny, nz = vol.shape
        xi = np.interp(verts_ijk[:, 0], np.arange(nx), x_lin)
        yi = np.interp(verts_ijk[:, 1], np.arange(ny), y_lin)
        zi = np.interp(verts_ijk[:, 2], np.arange(nz), z_lin)
        verts_xyz = np.column_stack([xi, yi, zi])
    
        cm = plt.get_cmap(cmap)
        facecolor = cm(norm(level))
    
        mesh = Poly3DCollection(verts_xyz[faces], facecolor=facecolor, edgecolor='none', alpha=alpha)
        ax.add_collection3d(mesh)
        return mesh
    
    def add_interpolated_volume_like_grid(grid_data):
        # add plotly-like volume via stacked isosurfaces 
        # input
        # grid_data is a dict-of-dicts containing grid points and scalar values
        # output
        # returns nothing
    
        x, y, z, vals_raw = get_xyz_and_vals(grid_data, is_exp=False)
        if x is None or vals_raw is None:
            return
    
        # decide which scalar field we interpolate + iso in
        # shared_axis: raw values
        # normalized_axis: normalized values in [0,1]
        if normalization_mode == "normalized_axis":
            vmin = float(np.min(grid_vals)) if len(grid_vals) else 0.0
            vmax = float(np.max(grid_vals)) if len(grid_vals) else 1.0
    
            vals_field = normalize_vals(vals_raw, vmin, vmax)
    
            lt = normalize_vals(np.array([lower_threshold], dtype=float), vmin, vmax)[0] if lower_threshold is not None else 0.0
            ut = normalize_vals(np.array([upper_threshold], dtype=float), vmin, vmax)[0] if upper_threshold is not None else 1.0
        else:
            vals_field = vals_raw
            lt = float(lower_threshold) if lower_threshold is not None else float(np.nanmin(vals_raw[np.isfinite(vals_raw)]))
            ut = float(upper_threshold) if upper_threshold is not None else float(np.nanmax(vals_raw[np.isfinite(vals_raw)]))
    
        # build inside-mask in the SAME space as vals_field
        valid = np.isfinite(vals_field) & np.isfinite(vals_raw) & (vals_raw != 0)
        inside = valid & (vals_field >= lt) & (vals_field <= ut)
    
        if np.sum(inside) < 10:
            return
    
        pts = np.column_stack((x[inside], y[inside], z[inside]))
        vals = vals_field[inside]
    
        # interpolate on regular 3D grid
        n_interp = 60
        x_lin = np.linspace(float(np.min(x)), float(np.max(x)), n_interp)
        y_lin = np.linspace(float(np.min(y)), float(np.max(y)), n_interp)
        z_lin = np.linspace(float(np.min(z)), float(np.max(z)), n_interp)
        X, Y, Z = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
    
        vol = griddata(pts, vals, (X, Y, Z), method='linear')
        if vol is None or np.all(~np.isfinite(vol)):
            return
    
        # fill nans so marching cubes works
        cmin = float(np.nanmin(vals))
        cmax = float(np.nanmax(vals))
        fill_val = cmin - 10.0 * (cmax - cmin + 1e-9)
        vol_filled = np.nan_to_num(vol, nan=fill_val)
    
        vmin_f = float(np.min(vol_filled))
        vmax_f = float(np.max(vol_filled))
        
        iso_min = max(float(lt), vmin_f)
        iso_max = min(float(ut), vmax_f)
        if not (iso_min < iso_max):
            return
        
        surface_count = 12
        levels = np.linspace(iso_min, iso_max, surface_count)
        
        for lvl in levels:
            plot_isosurface_from_volume(ax=ax, vol=vol_filled, x_lin=x_lin, y_lin=y_lin, z_lin=z_lin, level=lvl, norm=norm, cmap=cmap, alpha=(0.6 / surface_count))
        
    # translate exp axis keys and exp value key from grid reference
    exp_axes = translate_axes_grid_to_exp(grid_axes)
    exp_value_key = translate_value_key_grid_to_exp(value_key)

    # determine whether experimental data can be plotted in that space
    plot_exp = (None not in exp_axes)

    # enforce your original rules on exp value key usage
    if value_key in ('CV_V_m', 'CV_ISI', 'CV_ISI_per_energy'):
        exp_value_key = None
    if value_key in ('MI_calculated_bits_list', 'MI_calculated_per_energy_list', 'FI_calculated_list', 'FI_FZ_list'):
        value_key = None
        exp_value_key = None
    
    # compute shared normalization bounds
    grid_vals = []
    exp_vals = []
    if value_key is not None:
        grid_vals += collect_valid_vals(grid_data_exc, value_key)
        grid_vals += collect_valid_vals(grid_data_inh, value_key)
    if exp_value_key is not None and plot_exp:
        exp_vals += collect_valid_vals(exp_data_exc, exp_value_key)
        exp_vals += collect_valid_vals(exp_data_inh, exp_value_key)

    all_vals = []
    all_vals += grid_vals
    all_vals += exp_vals

    # set global default thresholds like your grid plot
    if len(all_vals) == 0:
        data_min, data_max = 0.0, 0.0
    else:
        data_min, data_max = float(np.min(all_vals)), float(np.max(all_vals))
    if lower_threshold is None:
        lower_threshold = data_min
    if upper_threshold is None:
        upper_threshold = data_max
    if lower_threshold > upper_threshold:
        lower_threshold, upper_threshold = upper_threshold, lower_threshold

    # choose label strings like in your correlation plotting
    axis_labels_axis = tuple(value_key_text_plot_Zeldenrust(a, plot_mode="correlation") for a in grid_axes)
    grid_value_label = value_key_text_plot_Zeldenrust(value_key, plot_mode='correlation_short') if value_key is not None else None
    exp_value_label = value_key_text_plot_Zeldenrust(value_key, plot_mode='correlation_short') if exp_value_key is not None else None

    # choose global colormap normalization
    if (value_key is not None) or (exp_value_key is not None):
        if normalization_mode == "normalized_axis":
            norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
        else:
            norm = Normalize(vmin=data_min, vmax=data_max, clip=True)
    else:
        norm = None

    
    outside_opacity_factor = 0.2

    # plot experimental points first, grid interpolation later, then experimental points again if you want them on top
    x, y, z, vals_raw = get_xyz_and_vals(exp_data_exc, is_exp=True)
    add_scatter(x, y, z, vals_raw, kind='exp', celltype='exc', base_color=color_exc, size=exp_marker_size, base_opacity=0.7)

    x, y, z, vals_raw = get_xyz_and_vals(exp_data_inh, is_exp=True)
    add_scatter(x, y, z, vals_raw, kind='exp', celltype='inh', base_color=color_inh, size=exp_marker_size, base_opacity=0.7)

    # interpolation mechanism 
    if interpolation is True:
        add_interpolated_volume_like_grid(grid_data_exc)
        add_interpolated_volume_like_grid(grid_data_inh)
    else:
        x, y, z, vals_raw = get_xyz_and_vals(grid_data_exc, is_exp=False)
        add_scatter(x, y, z, vals_raw, kind='grid', celltype='exc', base_color=color_exc, size=grid_marker_size, base_opacity=0.45)

        x, y, z, vals_raw = get_xyz_and_vals(grid_data_inh, is_exp=False)
        add_scatter(x, y, z, vals_raw, kind='grid', celltype='inh', base_color=color_inh, size=grid_marker_size, base_opacity=0.45)

    # re-plot exp to sit on top of volume if interpolation is enabled
    if interpolation is True:
        x, y, z, vals_raw = get_xyz_and_vals(exp_data_exc, is_exp=True)
        add_scatter(x, y, z, vals_raw, kind='exp', celltype='exc', base_color=color_exc, size=exp_marker_size, base_opacity=0.9)

        x, y, z, vals_raw = get_xyz_and_vals(exp_data_inh, is_exp=True)
        add_scatter(x, y, z, vals_raw, kind='exp', celltype='inh', base_color=color_inh, size=exp_marker_size, base_opacity=0.9)

    # ticks and label spacing 
    ax.tick_params(axis='x', pad=0)
    ax.tick_params(axis='y', pad=0)
    ax.tick_params(axis='z', pad=0)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
    
    ax.set_xlabel(axis_labels_axis[0], labelpad=0)
    ax.set_ylabel(axis_labels_axis[1], labelpad=0)
    ax.set_zlabel(axis_labels_axis[2], labelpad=-2)

    # make panes white
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # add colorbar
    if colorbar_mode and ((value_key is not None) or (exp_value_key is not None)) and (norm is not None):
        label = grid_value_label if grid_value_label is not None else exp_value_label
        if normalization_mode == "normalized_axis" and label is not None:
            label = f"{label} (normalized)"
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(np.array(all_vals, dtype=float) if len(all_vals) else np.array([0.0, 1.0]))
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.7, pad=0.2, aspect=20)
        if label is not None:
            cbar.set_label(label)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

    # set view
    ax.view_init(elev=elev, azim=azim)

    # save figure if requested
    if standalone and savename is not None:
        fig.savefig(f"../Figures/{savename}.pdf", bbox_inches="tight", dpi=dpi)
        fig.savefig(f"../Figures/{savename}.png", bbox_inches="tight", dpi=dpi)

    if standalone:
        plt.show()




###### correlation plotting functions Zeldenrust #############

def load_data_correlations_Zeldenrust(results, r_post_upper_bound=None):
    # load data for correlation plots (Zeldenrust grid results)
    #
    # input
    # results is the dictionary of saved values (grid dict-of-dicts)
    # r_post_upper_bound is a an optional upper firing rate bound to mask the data
    
    # output
    # pc1, pc2, pc3, E_L, V_thresh, R_m, C_m, Delta_T_ad, V_reset, tau_w_ad, a_ad, b_ad,
    # V_m, I_syn_e, I_syn_i, r_post, E_tot, CV_V_m, CV_ISI, CV_ISI_per_energy,
    # MI, MI_per_energy, MICE, MICE_per_energy, MI_vec, hit_fraction, false_alarm_fraction
    # are the masked correlation values

    # initialize lists
    pc1, pc2, pc3, E_L, V_thresh, R_m, C_m, Delta_T_ad, V_reset, tau_w_ad, a_ad, b_ad, V_m, I_syn_e, I_syn_i, r_post, E_tot, CV_V_m, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, MICE, MICE_per_energy, MI_vec, hit_fraction, false_alarm_fraction = [], [], [], [], [], [], [],  [], [], [], [], [],  [], [], [],  [], [], [],  [], [], [], [], [],  [], [], [], []

    # extract values from the dictionary
    for key in results:
        # --- axes / parameters (grid naming) ---
        pc1.append(results[key]['pc1'])
        pc2.append(results[key]['pc2'])
        pc3.append(results[key]['pc3'])

        E_L.append(results[key]['E_L'])
        V_thresh.append(results[key]['V_thresh'])
        R_m.append(results[key]['R_m'])
        C_m.append(results[key]['C_m'])
        Delta_T_ad.append(results[key]['Delta_T_ad'])
        V_reset.append(results[key]['V_reset'])
        tau_w_ad.append(results[key]['tau_w_ad'])
        a_ad.append(results[key]['a_ad'])
        b_ad.append(results[key]['b_ad'])
        
        V_m.append(results[key]['V_m'])
        I_syn_e.append(results[key]['I_syn_e'])
        I_syn_i.append(results[key]['I_syn_i'])
        r_post.append(results[key]['r_post'])
        E_tot.append(results[key]['E_tot'])
        CV_V_m.append(results[key]['CV_V_m'])
        CV_ISI.append(results[key]['CV_ISI'])
        CV_ISI_per_energy.append(results[key]['CV_ISI_per_energy'])
        MI.append(results[key]['MI'])
        MI_per_energy.append(results[key]['MI_per_energy'])
        MICE.append(results[key]['MICE'])
        MICE_per_energy.append(results[key]['MICE_per_energy'])
        #MI_vec.append(results[key]['MI_vec'])
        hit_fraction.append(results[key]['hit_fraction'])
        false_alarm_fraction.append(results[key]['false_alarm_fraction'])

    # filter out all 0.0 values & convert lists to numpy arrays for that
    #mask = np.array(r_post) != 0.0
    
    # convert r_post once
    r_post_arr = np.array(r_post, dtype=float)
    mask = (r_post_arr != 0.0) & np.isfinite(r_post_arr)

    # optional upper bound
    if r_post_upper_bound is not None:
        mask &= (r_post_arr < r_post_upper_bound)

    pc1 = np.array(pc1)[mask]
    pc2 = np.array(pc2)[mask]
    pc3 = np.array(pc3)[mask]

    E_L = np.array(E_L)[mask]
    V_thresh = np.array(V_thresh)[mask]
    R_m = np.array(R_m)[mask]
    C_m = np.array(C_m)[mask]
    Delta_T_ad = np.array(Delta_T_ad)[mask]
    V_reset = np.array(V_reset)[mask]
    tau_w_ad = np.array(tau_w_ad)[mask]
    a_ad = np.array(a_ad)[mask]
    b_ad = np.array(b_ad)[mask]

    V_m = np.array(V_m)[mask]
    I_syn_e = np.array(I_syn_e)[mask]
    I_syn_i = np.array(I_syn_i)[mask]
    r_post = np.array(r_post)[mask]
    E_tot = np.array(E_tot)[mask]
    CV_V_m = np.array(CV_V_m)[mask]
    CV_ISI = np.array(CV_ISI)[mask]
    CV_ISI_per_energy = np.array(CV_ISI_per_energy)[mask]
    MI = np.array(MI)[mask]
    MI_per_energy = np.array(MI_per_energy)[mask]
    MICE = np.array(MICE)[mask]
    MICE_per_energy = np.array(MICE_per_energy)[mask]
    #MI_vec = np.array(MI_vec, dtype=object)[mask] # keep MI_vec as object array (ragged / list entries)
    hit_fraction = np.array(hit_fraction)[mask]
    false_alarm_fraction = np.array(false_alarm_fraction)[mask]

    return pc1, pc2, pc3, E_L, V_thresh, R_m, C_m, Delta_T_ad, V_reset, tau_w_ad, a_ad, b_ad, V_m, I_syn_e, I_syn_i, r_post, E_tot, CV_V_m, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, MICE, MICE_per_energy, hit_fraction, false_alarm_fraction #MI_vec


def OLD_load_data_correlations_Zeldenrust(results):
    # load data for old correlation plots

    # input 
    # results is the dictionary of saved values

    # output
    # R_m, V_thresh, tau_w_ad, r_post, CV_V_m, I_syn_e, E_tot, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, MICE, MICE_per_energy are the masked correlation values

    
    # initialize lists
    R_m, V_thresh, tau_w_ad = [], [], []
    E_tot, r_post, V_m, CV_V_m, I_syn_e, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, MICE, MICE_per_energy, FI = [], [], [], [], [], [], [], [], [], [], [], []
    
    # extract values from the dictionary
    for key in results:
        
        R_m.append(float(key.split('_')[1]))
        V_thresh.append(float(key.split('_')[3]))
        tau_w_ad.append(float(key.split('_')[5]))

        E_tot.append(results[key]['E_tot'])
        r_post.append(results[key]['r_post'])
        V_m.append(results[key]['V_m'])
        CV_V_m.append(results[key]['CV_V_m'])
        I_syn_e.append(results[key]['I_syn_e'])
        CV_ISI.append(results[key]['CV_ISI'])
        CV_ISI_per_energy.append(results[key]['CV_ISI_per_energy'])
        MI.append(results[key]['MI'])
        MI_per_energy.append(results[key]['MI_per_energy'])
        MICE.append(results[key]['MICE'])
        MICE_per_energy.append(results[key]['MICE_per_energy'])
        FI.append(results[key]['MI_vec'][7])
        
    # filter out all 0.0 values & convert lists to numpy arrays for that
    mask = np.array(r_post) != 0.0
    
    R_m = np.array(R_m)[mask]
    V_thresh = np.array(V_thresh)[mask]
    tau_w_ad = np.array(tau_w_ad)[mask]
    
    E_tot = np.array(E_tot)[mask]
    r_post = np.array(r_post)[mask]
    V_m = np.array(V_m)[mask]
    CV_V_m = np.array(CV_V_m)[mask]
    I_syn_e = np.array(I_syn_e)[mask]
    CV_ISI = np.array(CV_ISI)[mask]
    CV_ISI_per_energy = np.array(CV_ISI_per_energy)[mask]
    MI = np.array(MI)[mask]
    MI_per_energy = np.array(MI_per_energy)[mask]
    MICE = np.array(MICE)[mask]
    MICE_per_energy = np.array(MICE_per_energy)[mask]
    FI = np.array(FI)[mask]
    
    
    return R_m, V_thresh, tau_w_ad, r_post, CV_V_m, I_syn_e, E_tot, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, MICE, MICE_per_energy, FI
    
############### Zeldenrust FR helping functions ###############

def add_banana_strip(ax, x0=210, x1=345, y_top=-58, y_bottom=-70, power=1.7, lw=28, color="k", alpha=0.9, zorder=5):
    # draw a thick shaded band as a single smooth curve with round caps
    
    xs = np.linspace(x0, x1, 250)
    t = (xs - x0) / (x1 - x0)          # 0..1
    ys = y_top + (y_bottom - y_top) * (t ** power)

    ax.plot(xs, ys, color=color, alpha=alpha, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=zorder)


def plot_gradient_line(ax, x, y, lw=3,  alpha_left=0.5, alpha_right=1.0, color="red", zorder=3):
    # plot a line with gradient for FR trajectory Zeldenrust
    x = np.asarray(x)
    y = np.asarray(y)

    # create line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # normalized position along x (0 → 1)
    t = (x[:-1] - x.min()) / (x.max() - x.min())

    # alpha gradient
    alphas = alpha_left + t * (alpha_right - alpha_left)

    # RGBA colors per segment
    base_rgb = to_rgba(color)[:3]
    colors = [(base_rgb[0], base_rgb[1], base_rgb[2], a) for a in alphas]

    lc = LineCollection(segments, colors=colors, linewidths=lw, zorder=zorder, capstyle="round", joinstyle="round")

    ax.add_collection(lc)
    ax.autoscale_view()
    