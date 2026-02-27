# paper plotting functions

# import predefined functions
import Functions.plotting_functions as pf
import Functions.analysis_functions as af

import os
import io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from PIL import Image
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import LogLocator
    
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns



def set_paper_style():
    # set global matplotlib style for paper-ready, square figures

    mpl.rcParams.update({
        "figure.figsize": (5.0, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 7,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"], # "CMU Serif"
    })
    # costum letter sizes
    panelletterfontsize=15
    
    return panelletterfontsize
    
def OLD_set_paper_style():
    # set global matplotlib style for paper-ready, square figures

    mpl.rcParams.update({
        "figure.figsize": (5.0, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"], # "CMU Serif"
    })
    # costum letter sizes
    panelletterfontsize=15
    
    return panelletterfontsize


def set_thesis_style():
    # set global matplotlib style for thesis figures (slightly larger text)

    mpl.rcParams.update({
        "figure.figsize": (5.0, 5.0),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.transparent": True,
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans", "Arial"], # "CMU Serif"
    })
    # costum letter sizes
    panelletterfontsize=14
    
    return panelletterfontsize

############################ helper functions ############################

def panel_letter(ax, letter, size=15, dx=-0.02, dy=0.02):
    fig = ax.figure
    bbox = ax.get_position()  # in figure coordinates
    fig.text(bbox.x0 + dx, bbox.y1 + dy, letter, fontsize=size, fontweight='bold', ha='left', va='bottom')

def render_3Dfig_to_ax(fig, ax, scale=2, trim_pad=1):
    # render a plotly figure to an axis object
    # input
    # ax is the given axis
    # fig is the given fig
    # scale is the scale to import the figure
    # trim_pad is the frame distance to cut the figure
    
    
    # tight layout & transparent background
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict( xaxis=dict(showbackground=False, visible=True), yaxis=dict(showbackground=False, visible=True), zaxis=dict(showbackground=False, visible=True)))

    # render png in memory
    png_bytes = pio.to_image(fig, format="png", scale=scale)
    im = Image.open(io.BytesIO(png_bytes)).convert("RGBA")

    # crop transparent borders
    alpha = im.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        L, T, R, B = bbox
        L = max(0, L - trim_pad)
        T = max(0, T - trim_pad)
        R = min(im.width,  R + trim_pad)
        B = min(im.height, B + trim_pad)
        im = im.crop((L, T, R, B))
        
    # show figure
    ax.imshow(np.asarray(im), interpolation='none')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.axis('off')

# Fig. 1
def fig_Pareto_optimality_Padamsey(results_mean_spiking_trials, exp_data=None, results_grid_point_to_exp_data=None, colors=['black', 'red', 'yellow'], fontsizes={'panelletterfontsize': 15}, figsize=(9,4.5), savename_mode=True): 
    # create Pareto_optimality figure
    # input
    # figA, figB, figC are the 3D figures
    # results_mean_spiking_trials is a dictionary of the full grid results
    # exp_data is a tuple of experimental data
    # results_grid_point_to_exp_data is a dictionary of the experimental data points fitted to their closest grid point 
    # colors are the colors for CTR, FR & highlight points
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    #color_CTR, color_FR, color_highlight = colors[0], colors[1], colors[2]
    
    fig = plt.figure(figsize=figsize, dpi=300) # 600
    #G = gs.GridSpec(2, 4, figure=fig, width_ratios=[0.04,1,1,1], height_ratios=[1.5,1], wspace=0.8, hspace=0.2) # wspace=0.45, hspace=0.3
    
    G_outer = gs.GridSpec(2, 2, figure=fig, width_ratios=[1, 0.01], height_ratios=[1.3, 0.7], wspace=0.2, hspace=0.2)
    
    # A–C 
    G_top = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=G_outer[0, 0], wspace=0.3) # tight spacing
    axA=fig.add_subplot(G_top[0, 0], projection='3d')
    pf.plot_grid_3D_matplotlib(results=results_mean_spiking_trials, value_key=None, lower_threshold=0.45, upper_threshold=None, colorbar_mode=False, interpolation=True, all_trajectories=None, legend_mode=False, axes_mode=False, exp_data=exp_data, plot_exp_stems=True, plot_ellipsoids=True, elev=20, azim=130, ax=axA) #dpi=1000,
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'], dx=0.0, dy=-0.06)
    
    axB=fig.add_subplot(G_top[0, 1], projection='3d')
    pf.plot_grid_3D_matplotlib(results=results_mean_spiking_trials, value_key="OSI", lower_threshold=0.01, upper_threshold=None, colorbar_mode=False, interpolation=False, all_trajectories=None, legend_mode=False, axes_mode=False, exp_data=None, plot_exp_stems=True, plot_ellipsoids=True, elev=20, azim=130, ax=axB) #dpi=1000,
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'], dx=0.0, dy=-0.06)
    
    axC=fig.add_subplot(G_top[0, 2], projection='3d')
    # plot colobar outside the plot
    caxC = fig.add_subplot(G_outer[0, 1])
    pos = caxC.get_position()
    new_height = pos.height * 0.6
    new_y0 = pos.y0 + (pos.height - new_height) / 2
    caxC.set_position([pos.x0, new_y0, pos.width, new_height])
    #caxC = inset_axes(axC, width="4%", height="70%", loc="center left", bbox_to_anchor=(1.08, 0., 1, 1), bbox_transform=axC.transAxes, borderpad=0)
    pf.plot_grid_3D_matplotlib(results=results_mean_spiking_trials, value_key="OSI", lower_threshold=0.45, upper_threshold=None, colorbar_mode=True, interpolation=True, all_trajectories=None, legend_mode=False, axes_mode=False, exp_data=exp_data, plot_exp_stems=True, plot_ellipsoids=True, elev=20, azim=130, cax=caxC, ax=axC) #dpi=1000,
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'], dx=0.0, dy=-0.06)
    
    # smaller space
    #axspace=axes[3:6]
    
    # D -- I
    G_bottom = gs.GridSpecFromSubplotSpec(1, 6, subplot_spec=G_outer[1,:], width_ratios=[0.1, 1, 0.1, 1, 0.1, 1], wspace=0.5) # larger white space
    R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy, MI_new, MI_new_per_energy, MICE_new, MICE_new_per_energy, TE_new, TE_new_per_energy, TECE_new, TECE_new_per_energy  = pf.load_data_correlations(results_mean_spiking_trials)
    #R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy = pf.load_data_correlations(results_mean_spiking_trials)

    # D
    axD=fig.add_subplot(G_bottom[0, 1])
    pf.plot_correlation(E_tot, OSI, 'E_tot', 'OSI', OSI_per_energy, 'OSI_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, inverted_x=True, colors=colors, plot_mode='correlation_short', ax=axD)
    axD.xaxis.get_offset_text().set_y(20.0)
    axDlabel=fig.add_subplot(G_bottom[0, 0])
    axDlabel.axis("off")
    panel_letter(axDlabel, "D", size=fontsizes['panelletterfontsize'], dx=0.0, dy=0.01)
    
    # E
    axE=fig.add_subplot(G_bottom[0, 3])
    pf.plot_correlation(r_post, MICE, 'r_post', 'MICE', MICE_per_energy, 'MICE_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, colors=colors, plot_mode='correlation_short', ax=axE)
    axE.set_xlim(left=-0.5, right=10) 
    axE.get_legend().remove()
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'], dx=-0.06, dy=0.01)

    # F
    axF=fig.add_subplot(G_bottom[0, 5])
    #fit_func_highlight=pf.linear_rate_information_func # options: linear_func 2, sqrt_func 2, log_func 3, rate_information_func 3, linear_rate_information_func 5, sqrt_rate_information_func 5, linear_log_func 4, piecewise_linear_exponential_func 3
    #initial_guess_highlight=[1,1,1,1,1]
    
    #R_m_init, E_L_init = 150.0, -50.0
    #x_label, y_label = 'r_post', 'MI_per_energy'
    
    #w_scale_list_x, x_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, x_label, R_m_init, E_L_init)
    #w_scale_list_y, y_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, y_label, R_m_init, E_L_init)
        
    #highlight_points = (x_highlights, y_highlights)
    
    # z: E_tot, 'E_tot'
    #pf.plot_correlation(r_post, MI_per_energy, x_label, y_label, highlight_points=highlight_points, fit_func_highlight=fit_func_highlight, initial_guess_highlight=initial_guess_highlight, colors=colors, plot_mode='correlation_short', ax=axF)
    pf.plot_correlation(r_post, MI_per_energy, 'r_post', 'MI_per_energy', E_tot, 'E_tot', results_grid_point_to_exp_data=results_grid_point_to_exp_data, colors=colors, plot_mode='correlation_short', ax=axF)
    axF.get_legend().remove()
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'], dx=-0.03, dy=0.01)
    
    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_1_Pareto_optimality_Padamsey.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True, pad_inches=0.2, dpi=1800)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=600, bbox_inches='tight', transparent=True, pad_inches=0.2)


#Fig. 2
def fig_Pareto_optimality_Zeldenrust(results_analysis_exc, results_analysis_inh, results_analysis_PCA_analyzed_exc, results_analysis_PCA_analyzed_inh, results_PC_exc, results_PC_inh, proportion_of_synaptic_change, MI_list, E_tot_list, colors=['coral', 'cornflowerblue', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(9,9), savename_mode=True): 
    # create Pareto optimality figure for Zeldenrust data
    # input
    # results_analysis_exc & results_analysis_inh are dictionaries of the experimental data points 
    # results_analysis_PCA_analyzed_exc & results_analysis_PCA_analyzed_inh is a results dict (dict-of-dicts) for experimental data projected into PCA space
    # results_PC_exc, results_PC_inh are dictionaries of the PC grids
    # proportion_of_synaptic_change, MI_list, E_tot_list are the lists of predicted FR trajectories for Zeldenrust data
    # colors are the colors for excitatory & inhibitory plots & color_FR
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    color_exc, color_inh, color_FR = colors[0], colors[1], colors[2]
    fig = plt.figure(figsize=figsize, dpi=600)
    G = gs.GridSpec(4, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1.8, 1.0, 0.1, 1.0], wspace=1.0, hspace=0.25)
    
    # A -- B 
    row1 = G[0, :].subgridspec(1, 2, wspace=0.1)
    
    # A
    #axA = fig.add_subplot(G[0, 0:3], projection='3d')
    #render_3Dfig_to_ax(figA, axA)
    axA = fig.add_subplot(row1[0, 0], projection='3d')
    pf.plot_grid_and_exp_3D_matplotlib(grid_data_exc=results_PC_exc, grid_data_inh=None, exp_data_exc=results_analysis_PCA_analyzed_exc, exp_data_inh=None, value_key='MI', normalization_mode="normalized_axis", grid_axes=('b_ad', 'tau_w_ad', 'a_ad'), interpolation=True, lower_threshold=None, upper_threshold=None, color_exc='coral', color_inh='darkcyan', grid_marker_size=4, exp_marker_size=6, title=None, colorbar_mode=True, legend_mode=False, axes_mode=False, elev=20, azim=140, ax=axA)
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'], dx=0.0, dy=-0.06)
    
    # B
    #axB = fig.add_subplot(G[0, 3:6], projection='3d')
    #render_3Dfig_to_ax(figB, axB)
    axB = fig.add_subplot(row1[0, 1], projection='3d')
    pf.plot_grid_and_exp_3D_matplotlib(grid_data_exc=None, grid_data_inh=results_PC_inh, exp_data_exc=None, exp_data_inh=results_analysis_PCA_analyzed_inh, value_key='MI', normalization_mode="normalized_axis", grid_axes=('R_m', 'tau_w_ad', 'V_thresh'), interpolation=True, lower_threshold=None, upper_threshold=None, color_exc='coral', color_inh='darkcyan', grid_marker_size=4, exp_marker_size=6, title=None, colorbar_mode=True, legend_mode=False, axes_mode=False, elev=20, azim=110, ax=axB)
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'], dx=0.0, dy=-0.06)
    
    # C: excitatory Pareto
    row2 = G[1, :].subgridspec(1, 3, wspace=0.5) # extra width space

    axC = fig.add_subplot(row2[0, 0])
    #axC = fig.add_subplot(G[1, 0:2])
    pc1_grid_exc, pc2_grid_exc, pc3_grid_exc, E_L_grid_exc, V_thresh_grid_exc, R_m_grid_exc, C_m_grid_exc, Delta_T_ad_grid_exc, V_reset_grid_exc, tau_w_ad_grid_exc, a_ad_grid_exc, b_ad_grid_exc, V_m_grid_exc, I_syn_e_grid_exc, I_syn_i_grid_exc, r_post_grid_exc, E_tot_grid_exc, CV_V_m_grid_exc, CV_ISI_grid_exc, CV_ISI_per_energy_grid_exc, MI_grid_exc, MI_per_energy_grid_exc, MICE_grid_exc, MICE_per_energy_grid_exc, hit_fraction_grid_exc, false_alarm_fraction_grid_exc = pf.load_data_correlations_Zeldenrust(results_PC_exc, r_post_upper_bound=100)
    
    x_exp_name = 'E_tot_1e9_ATP_per_s_list'
    y_exp_name = 'MI_FZ_bits_list' 
    #z_exp_name = 'MI_FZ_per_energy_list' 
    
    x_grid_exc = np.asarray(E_tot_grid_exc)/1e9
    y_grid_exc = np.asarray(MI_grid_exc)
    z_grid_exc = np.asarray(MI_per_energy_grid_exc)
    
    x_min_grid_exc = min(x_grid_exc)
    y_max_grid_exc = max(y_grid_exc)
    z_max_grid_exc = max(z_grid_exc)
    
    pf.plot_correlation(x_grid_exc/x_min_grid_exc, y_grid_exc/y_max_grid_exc, 'normalized $E_{tot}$', 'normalized $MI$', z=z_grid_exc/z_max_grid_exc, z_label='norm. $MI$ per energy', ax=axC) #, inverted_x=True
    
    x_exp_exc = np.asarray(results_analysis_exc[x_exp_name])
    y_exp_exc = results_analysis_exc[y_exp_name]
    #z_exp_exc = results_analysis_exc[z_exp_name]
    
    x_min_exp_exc = min(np.asarray(x_exp_exc))
    y_max_exp_exc = max(np.asarray(y_exp_exc))
    #z_max_exp_exc = max(np.asarray(z_exp_exc))

    pf.plot_correlation_exc_inh(x_exp_exc/x_min_exp_exc, [], y_exp_exc/y_max_exp_exc, [], 'normalized $E_{tot}$', 'normalized $MI$', inverted_x=True, colors=[color_exc, color_inh], transparency=0.7, ax=axC)

    axC.get_legend().remove()
    """
    axC.set_xscale('log')
    xmin, xmax = axC.get_xlim()
    ticks = 10 ** np.linspace(np.log10(xmin), np.log10(xmax),3)
    from matplotlib.ticker import LogFormatterMathtext, NullLocator, NullFormatter
    axC.set_xticks(ticks)
    axC.xaxis.set_major_formatter(LogFormatterMathtext())

    # kill minor ticks + labels completely
    axC.xaxis.set_minor_locator(NullLocator())
    axC.xaxis.set_minor_formatter(NullFormatter())
    """
    """
    handles, labels = axC.get_legend_handles_labels()
    exc_handles = [h for h, l in zip(handles, labels) if "exc" in l.lower()]
    exc_labels  = [l for l in labels if "exc" in l.lower()]
    axC.legend(exc_handles, exc_labels, frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', loc='upper left')
    """
    
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    
    # D: inhibitory Pareto
    axD = fig.add_subplot(row2[0, 1])
    #axD = fig.add_subplot(G[1, 2:4])
    pc1_grid_inh, pc2_grid_inh, pc3_grid_inh, E_L_grid_inh, V_thresh_grid_inh, R_m_grid_inh, C_m_grid_inh, Delta_T_ad_grid_inh, V_reset_grid_inh, tau_w_ad_grid_inh, a_ad_grid_inh, b_ad_grid_inh, V_m_grid_inh, I_syn_e_grid_inh, I_syn_i_grid_inh, r_post_grid_inh, E_tot_grid_inh, CV_V_m_grid_inh, CV_ISI_grid_inh, CV_ISI_per_energy_grid_inh, MI_grid_inh, MI_per_energy_grid_inh, MICE_grid_inh, MICE_per_energy_grid_inh, hit_fraction_grid_inh, false_alarm_fraction_grid_inh = pf.load_data_correlations_Zeldenrust(results_PC_inh, r_post_upper_bound=100)

    x_grid_inh = np.asarray(E_tot_grid_inh)/1e9
    y_grid_inh = np.asarray(MI_grid_inh)
    z_grid_inh = np.nan_to_num(np.asarray(MI_per_energy_grid_inh), nan=0.0)

    x_min_grid_inh =  min(x_grid_inh)
    y_max_grid_inh = max(y_grid_inh)
    z_max_grid_inh = max(z_grid_inh)

    pf.plot_correlation(x_grid_inh/x_min_grid_inh, y_grid_inh/y_max_grid_inh, 'normalized $E_{tot}$', 'normalized $MI$', z=z_grid_inh/z_max_grid_inh, z_label='norm. $MI$ per energy', ax=axD)  #, inverted_x=True

    x_exp_inh = np.asarray(results_analysis_inh[x_exp_name])
    y_exp_inh = results_analysis_inh[y_exp_name]
    #z_exp_inh = results_analysis_inh[z_exp_name]

    x_min_exp_inh = min(np.asarray(x_exp_inh))
    y_max_exp_inh = max(np.asarray(y_exp_inh))
    #z_max_exp_inh = max(np.asarray(z_exp_inh))

    pf.plot_correlation_exc_inh([], x_exp_inh/x_min_exp_inh, [], y_exp_inh/y_max_exp_inh, 'normalized $E_{tot}$', 'normalized $MI$', inverted_x=True, colors=[color_exc, color_inh], transparency=0.7, ax=axD)

    axD.get_legend().remove()
    """
    handles, labels = axD.get_legend_handles_labels()
    inh_handles = [h for h, l in zip(handles, labels) if "inh" in l.lower()]
    inh_labels  = [l for l in labels if "inh" in l.lower()]
    axD.legend(inh_handles, inh_labels, frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', loc='lower left')
    """
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])
    
    # E
    axE = fig.add_subplot(row2[0, 2])
    #axE = fig.add_subplot(G[1, 4:6])

    x_name = 'firing_rate_calculated_Hz_windowed'
    y_name_MICE_FZ = 'MICE_FZ_windowed_list'
    firing_rate_Hz_windowed_list_flat_exc = np.concatenate(results_analysis_exc[x_name]).tolist()
    MICE_FZ_bits_windowed_list_flat_exc = np.concatenate(results_analysis_exc[y_name_MICE_FZ]).tolist()
    firing_rate_Hz_windowed_list_flat_inh = np.concatenate(results_analysis_inh[x_name]).tolist()
    MICE_FZ_bits_windowed_list_flat_inh = np.concatenate(results_analysis_inh[y_name_MICE_FZ]).tolist()
    pf.plot_correlation_exc_inh(firing_rate_Hz_windowed_list_flat_exc, firing_rate_Hz_windowed_list_flat_inh, np.array(MICE_FZ_bits_windowed_list_flat_exc)*3*4, np.array(MICE_FZ_bits_windowed_list_flat_inh)*18*4, '$r_{post}$ (Hz)', '$CE_{MI}$ (bits/Hz)', colors=[color_exc, color_inh], ax=axE)
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'])
    
    # F
    row3 = G[3, :].subgridspec(1, 3, wspace=0.5) # extra width space
    
    axF = fig.add_subplot(row3[0, 0])
    x_name = 'R_m_mean_MOhm_list'
    y_name = 'tau_w_mean_ms_list'
    pf.plot_correlation_exc_inh(results_analysis_exc[x_name], results_analysis_inh[x_name], results_analysis_exc[y_name], results_analysis_inh[y_name], x_name, y_name, colors=[color_exc, color_inh], ax=axF)
    axF.get_legend().remove()
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])
    
    # G
    axG = fig.add_subplot(row3[0, 1])#fig.add_subplot(G[2, 1:3])
    x_name = 'R_m_mean_MOhm_list'
    y_name = 'E_L_mV_list' 
    
    pf.plot_correlation_exc_inh(results_analysis_exc[x_name], results_analysis_inh[x_name], results_analysis_exc[y_name], results_analysis_inh[y_name], x_name, y_name, colors=[color_exc, color_inh], ax=axG)
    pf.add_banana_strip(axG, x0=220, x1=350, y_top=-58, y_bottom=-70, power=3.8, lw=30, color=color_FR, alpha=0.4, zorder=10)
    if axG.get_legend() is not None:
        axG.get_legend().remove()
    
    panel_letter(axG, "G", size=fontsizes['panelletterfontsize'])
    
    # H: MI & E_tot FR
    hcell = row3[0, 2].subgridspec(2, 1, hspace=0.25, height_ratios=[1, 1])

    axH1 = fig.add_subplot(hcell[0, 0])  # MI
    axH2 = fig.add_subplot(hcell[1, 0])  # E_tot

    pf.plot_gradient_line(axH1, proportion_of_synaptic_change, MI_list, lw=4, alpha_left=0.3, alpha_right=1.0, color=color_FR, zorder=5)
    axH1.set_ylabel('$MI$ (bits)')
    axH1.set_xlabel("")
    axH1.tick_params(axis="x", labelbottom=False)
    axH1.spines['top'].set_visible(False)
    axH1.spines['right'].set_visible(False)

    pf.plot_gradient_line(axH2, proportion_of_synaptic_change, np.asarray(E_tot_list) / 1e9, lw=4, alpha_left=0.3, alpha_right=1.0, color=color_FR, zorder=5)
    axH2.set_xlabel('Prop. of FR syn. weight change')
    axH2.set_ylabel('$E_{tot}$ ($10^9$ATP/s)')
    axH2.spines['top'].set_visible(False)
    axH2.spines['right'].set_visible(False)

    # Panel letter für G (oben links im oberen Subplot)
    panel_letter(axH1, "H", size=fontsizes['panelletterfontsize'])
    
    #plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_2_Pareto_optimality_Zeldenrust.pdf"
        #plt.savefig(savepath, bbox_inches='tight', transparent=False, dpi=600)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=900, bbox_inches='tight', transparent=True)
        
    print('norm. const. $E_{tot}$ grid exc: ' + str(round(x_min_grid_inh,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ grid exc: ' + str(round(y_max_grid_inh,4)) + ' bits')
    print('norm. const. $MI$ per energy grid exc: ' + str(round(z_max_grid_exc,4)) + ' bits/($10^{9}$ ATP/s)') 
    
    print('norm. const. $E_{tot}$ exp exc: ' + str(round(x_min_exp_exc,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ exp exc: ' + str(round(y_max_exp_exc,4)) + ' bits') 
    #print('norm. const. $MI$ per energy exp exc: ' + str(round(z_max_exp_exc,4)) + ' bits/($10^{9}$ ATP/s)') 
    
    print('norm. const. $E_{tot}$ grid inh: ' + str(round(x_min_grid_inh,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ grid inh: ' + str(round(y_max_grid_inh,4)) + ' bits')  
    print('norm. const. $MI$ per energy grid exc: ' + str(round(z_max_grid_inh,4)) + ' bits/($10^{9}$ ATP/s)') 
    
    print('norm. const. $E_{tot}$ exp inh: ' + str(round(x_min_exp_inh,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ grid exc: ' + str(round(y_max_exp_inh,4)) + ' bits')
    #print('norm. const. $MI$ per energy exp exc: ' + str(round(z_max_exp_inh,4)) + ' bits/($10^{9}$ ATP/s)') 
        
# Fig. 3
def fig_energy_budget(E_CTR, E_FR, r_post_fit, colors=['black', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(7,6), savename_mode=True):
    # create energy budget figure
    
    # input
    # E_CTR = [E_tot_CTR, E_HK_CTR, E_RP_CTR, E_AP_CTR, E_ST_CTR, E_glu_CTR, E_Ca_CTR]
    # E_FR  = [E_tot_FR,  E_HK_FR,  E_RP_FR,  E_AP_FR,  E_ST_FR,  E_glu_FR,  E_Ca_FR]
    # r_post_fit is array of firing rates (Hz)
    # colors are the colors for color_CTR, color_FR
    # fontsizes is a dictionary of used font sizes
    # figsize is the figure size
    # savename_mode decides whether fig is saved or not
    
    color_CTR, color_FR = colors[0], colors[1]
    
    E_tot_CTR, E_HK_CTR, E_RP_CTR, E_AP_CTR, E_ST_CTR, E_glu_CTR, E_Ca_CTR = E_CTR
    E_tot_FR,  E_HK_FR,  E_RP_FR,  E_AP_FR,  E_ST_FR,  E_glu_FR,  E_Ca_FR  = E_FR

    r_post_optimum = 4.0  # Hz
    idx_optimum = (np.abs(r_post_fit - r_post_optimum)).argmin()

    description_CTR = 'CTR'
    description_FR  = 'FR'
    legend_pos = 'upper right'
    y_label = '$E_{tot}$ ($10^{9}$ ATP/s)'
    y_limit = 5.0

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1.0, 0.5, 0.5, 1.0], hspace=0.6, wspace=0.0)

    # top row: A spans cols 0–1, B spans cols 2–3
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4])

    # bottom row: C at col 0, D at col 3, middle cols are empty
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 3])
    
    """
    fig, axes = plt.subplots(2,4, figsize=figsize, gridspec_kw={'height_ratios': [1.0, 1.0], 'width_ratios': [1.0, 0.2, 0.2, 1.0], 'hspace': 0.6, 'wspace': 0.0})

    axA = axes[0, 0:2]
    axB = axes[0, 2:3]
    axC = axes[1, 0]
    axCspace = axes[1, 1]
    axDspace = axes[1, 2]
    axD = axes[1, 3]
    
    # switch off unused axes
    #axes[0, 0].axis('off')
    #axes[1, 0].axis('off')
    for j in range(2):
        axes[1, j].axis('off')"""
    
    legend_labels = ['House keeping', 'Resting potential\n(reversal of Na\u207A)', 'Action potential\n(reversal of Na\u207A)', 'Synaptic transmission\n(glutamate recycling)', 'Synaptic transmission\n(reversal of presyn Ca\u00B2\u207A)', 'Synaptic transmission\n(reversal of Na\u207A)']
    
    # panel A: CTR stackplot
    pf.plot_energy_stackplot(np.asarray(E_tot_CTR)/1e9, np.asarray(E_HK_CTR)/1e9, np.asarray(E_RP_CTR)/1e9, np.asarray(E_AP_CTR)/1e9, np.asarray(E_ST_CTR)/1e9, np.asarray(E_glu_CTR)/1e9, np.asarray(E_Ca_CTR)/1e9, np.asarray(r_post_fit), description_CTR, legend_labels, r_post_optimum, r_post_optimum_percentages=True, inverted=False, legend_pos=False, y_limit=y_limit, y_label=y_label, color_r_post_optimum=color_CTR, ax=axA)
    axA.set_title('CTR', fontweight='bold', color=color_CTR, fontsize=fontsizes['panelletterfontsize']*0.8)
    axA.text(-0.12, 1.04, "A", transform=axA.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    # panel B: FR stackplot (inverted x-axis, legend here)
    pf.plot_energy_stackplot(np.asarray(E_tot_FR)/1e9, np.asarray(E_HK_FR)/1e9, np.asarray(E_RP_FR)/1e9, np.asarray(E_AP_FR)/1e9, np.asarray(E_ST_FR)/1e9, np.asarray(E_glu_FR)/1e9, np.asarray(E_Ca_FR)/1e9, np.asarray(r_post_fit), description_FR, legend_labels, r_post_optimum, r_post_optimum_percentages=True, inverted=True, legend_pos=legend_pos, y_limit=y_limit, y_label=y_label, color_r_post_optimum=color_FR, ax=axB)
    axB.set_title('FR', fontweight='bold', color=color_FR, fontsize=fontsizes['panelletterfontsize']*0.8)
    #ppf.panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])
    #axB.text(-0.01, 1.04, "B", transform=axB.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    legB = axB.get_legend()
    
    # legend 
    axLegend = fig.add_subplot(gs[1, 1:3])  # spans the two middle columns
    axLegend.axis('off')
    if legB is not None:
        handles, labels = axB.get_legend_handles_labels()
        legB.remove()  # remove legend from B
        axLegend.legend(handles, labels, loc='center', frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, reverse=True)

    # panel C: CTR pie
    pf.plot_energy_pie_chart(E_HK_CTR[idx_optimum], E_RP_CTR[idx_optimum], E_AP_CTR[idx_optimum], E_ST_CTR[idx_optimum], E_glu_CTR[idx_optimum], E_Ca_CTR[idx_optimum], r_post_optimum, title_mode=False, label_mode='no', ax=axC)
    axC.text(-0.19, 1.04, "B", transform=axC.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    # panel D: FR pie
    insetD = inset_axes(axD, width="90%", height="90%", loc="center")
    pf.plot_energy_pie_chart(E_HK_FR[idx_optimum], E_RP_FR[idx_optimum], E_AP_FR[idx_optimum], E_ST_FR[idx_optimum], E_glu_FR[idx_optimum], E_Ca_FR[idx_optimum], r_post_optimum, title_mode=False, label_mode='no', ax=insetD)
    for spine in axD.spines.values():
        spine.set_visible(False)
    axD.set_xticks([])
    axD.set_yticks([])
    axD.text(-0.5, 1.04, "C", transform=axD.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    
    # add vertical + horizontal connector lines A -- C (black)
    x_disp_A, _ = axA.transData.transform((r_post_optimum, 0.0))
    x_fig_A, _ = fig.transFigure.inverted().transform((x_disp_A, 0.0))
    posA = axA.get_position()
    posC = axC.get_position()
    y_top_A = posA.ymin
    y_bar_C = posC.ymax + 0.07
    fig.add_artist(Line2D([x_fig_A, x_fig_A], [y_top_A, y_bar_C], transform=fig.transFigure, color=color_CTR, linestyle='--', linewidth=2.0, zorder=3))
    fig.add_artist(Line2D([posC.xmin, posC.xmax], [y_bar_C, y_bar_C], transform=fig.transFigure, color=color_CTR, linestyle='-', linewidth=3.0, zorder=3))

    # add vertical + horizontal connector lines B -- D (red)
    x_disp_B, _ = axB.transData.transform((r_post_optimum, 0.0))
    x_fig_B, _ = fig.transFigure.inverted().transform((x_disp_B, 0.0))
    posB = axB.get_position()
    posD = axD.get_position()
    y_top_B = posB.ymin
    y_bar_D = posD.ymax + 0.07
    fig.add_artist(Line2D([x_fig_B, x_fig_B], [y_top_B, y_bar_D], transform=fig.transFigure, color=color_FR, linestyle='--', linewidth=2.0, zorder=3))
    fig.add_artist(Line2D([posD.xmin, posD.xmax], [y_bar_D, y_bar_D], transform=fig.transFigure, color=color_FR, linestyle='-', linewidth=3.0, zorder=3))

    if savename_mode is True:
        savepath = '../Figures/0_paper/fig_3_energy_budget.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        
    plt.show()
    plt.close(fig)
  

  

# Fig. 4
def fig_multiple_trajectories_examples(results_mean_spiking_trials, all_trajectories, data_CTR_like, data_FR_like, colors=['#4a61e8', '#c41c5f', 'black'], fontsizes={'panelletterfontsize': 15}, figsize=(5,8), savename_mode=True):
    # compose the 3D + combined example traces + histogram figure
    #
    # input
    # results_mean_spiking_trials is the dictionary of results
    # all_trajectories are 20 randomly chosen trajectories from all
    # data_CTR_like, data_FR_like are dictionaries of the CTR_like & FR_like points
    # colors are the colors for color_CTR_like, color_FR_like, color_CTR
    # fontsizes is a dictionary of used font sizes
    # figsize is the figure size
    # savename_mode decides whether fig is saved or not
    

    # simulated V_m traces
    V_m_CTR_like = data_CTR_like['V_m']
    V_m_FR_like = data_FR_like['V_m']

    # shift V_m so resting level is at 0 mV
    V_m_shifted_CTR_like, _ = pf.shift_V_m(V_m_CTR_like, target_mV=0.0)
    V_m_shifted_FR_like, _ = pf.shift_V_m(V_m_FR_like, target_mV=0.0)

    # cut excerpts
    V_m_excerpts_CTR_like, V_m_min_CTR_like, V_m_max_CTR_like, _ = pf.cut_into_excerpts(V_m_shifted_CTR_like, n_excerpts=9, start_idx=3000, stop_idx=30000)
    V_m_excerpts_FR_like, V_m_min_FR_like, V_m_max_FR_like, _ = pf.cut_into_excerpts(V_m_shifted_FR_like, n_excerpts=9, start_idx=3000, stop_idx=30000)

    # shared y-limits
    V_m_min, V_m_max = pf.compute_shared_ylim(V_m_min_CTR_like, V_m_max_CTR_like, V_m_min_FR_like, V_m_max_FR_like)

    # create final figure (2 rows × 2 columns; A spans full top row)
    fig = plt.figure(figsize=figsize)
    G = gs.GridSpec(2, 2, figure=fig, height_ratios=[4.5, 1.5], wspace=0.20, hspace=0.08)
    #fig.subplots_adjust(top=0.98, bottom=0.14, left=0.06, right=0.99)

    # Panel A: 3D figure (spanning full upper row)
    #axA = fig.add_subplot(G[0, :])
    axA = fig.add_subplot(G[0, :], projection='3d')
    
    pf.plot_grid_3D_matplotlib(results=results_mean_spiking_trials, value_key="OSI_per_energy", lower_threshold=0.17, upper_threshold=None, colorbar_mode=True, interpolation=True, all_trajectories=all_trajectories, legend_mode=False, axes_mode=False, exp_data=None, plot_exp_stems=True, plot_ellipsoids=True, elev=20, azim=130, ax=axA) #dpi=1000,
    # add CTR-like & FR-like markers to the 3D fig
    pf.add_balls_to_3d_ax(axA, data_CTR_like, data_FR_like, colors=colors, size=180)
    
    color_CTR_like, color_FR_like, color_CTR = colors[0], colors[1], colors[2]
    # extract CTR-like and FR-like locations and convert w_scale → w_e
    R_m_CTR_like, E_L_CTR_like, w_scale_CTR_like = data_CTR_like['R_m'], data_CTR_like['E_L'], data_CTR_like['w_scale']
    R_m_FR_like, E_L_FR_like, w_scale_FR_like = data_FR_like['R_m'], data_FR_like['E_L'], data_FR_like['w_scale']

    w_e_CTR_like = float(pf.w_scale_to_w_e_syn(w_scale_CTR_like))
    w_e_FR_like = float(pf.w_scale_to_w_e_syn(w_scale_FR_like))
    
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=-0.07)
    
    # B: combined excerpts
    axB = fig.add_subplot(G[1, 0])
    pf.plot_two_V_m_excerpts(V_m_excerpts_1=V_m_excerpts_CTR_like, V_m_excerpts_2=V_m_excerpts_FR_like, color_1=color_CTR_like, color_2=color_FR_like, ylims=(V_m_min, V_m_max), ax=axB)
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    # add colored parameter text below panel B
    Bbox = axB.get_position()
    x_center = (Bbox.x0 + Bbox.x1) / 1.9
    y_base = Bbox.y0 - 0.05
    dy = 0.025
    x_label = x_center - 0.1
    x_CTR_like = x_center + 0.13
    x_FR_like = x_center + 0.01
    
    # R_m line
    fig.text(x_label, y_base + 2*dy, r'$R_m$:', ha='left', va='top')
    fig.text(x_FR_like, y_base + 2*dy, rf'${R_m_FR_like:.0f}\,\mathrm{{M\Omega}}$', ha='left', va='top', color=color_FR_like)
    fig.text(x_CTR_like, y_base + 2*dy, rf'${R_m_CTR_like:.0f}\,\mathrm{{M\Omega}}$', ha='left', va='top', color=color_CTR_like)
    
    # V_rest line
    fig.text(x_label, y_base + dy, r'$V_{\mathrm{rest}}$:', ha='left', va='top')
    fig.text(x_FR_like, y_base + dy, rf'${E_L_FR_like:.0f}\,\mathrm{{mV}}$', ha='left', va='top', color=color_FR_like)
    fig.text(x_CTR_like, y_base + dy, rf'${E_L_CTR_like:.0f}\,\mathrm{{mV}}$', ha='left', va='top', color=color_CTR_like)
    
    # <w_syn,e> line
    fig.text(x_label, y_base + 0*dy, r'$\langle w_{\mathrm{syn,e}}\rangle$:', ha='left', va='top')
    fig.text(x_FR_like, y_base + 0*dy, rf'${w_e_FR_like:.2f}\,\mathrm{{nS}}$', ha='left', va='top', color=color_FR_like)
    fig.text(x_CTR_like, y_base + 0*dy, rf'${w_e_CTR_like:.2f}\,\mathrm{{nS}}$', ha='left', va='top', color=color_CTR_like)
    
    # C: histogram
    xmin = -45
    xmax = -80
    V_thresh = data_CTR_like['V_thresh']

    axC = fig.add_subplot(G[1, 1])
    axC.hist(V_m_FR_like, bins=np.linspace(xmax, xmin, 80), color=color_FR_like, alpha=0.8, density=True)
    axC.hist(V_m_CTR_like, bins=np.linspace(xmax, xmin, 80), color=color_CTR_like, alpha=0.8, density=True)
    axC.axvline(V_thresh, 0, 0.85, color=color_CTR, ls='--', alpha=0.9)
    axC.text(V_thresh, 0.27, r'$V_{\mathrm{thresh}}$', ha='center', va='bottom')
    axC.set_xlim(xmin, xmax)
    axC.set_ylim(0, 0.3)
    axC.set_xlabel(r'$V_m$ (mV)')
    axC.yaxis.set_visible(False)
    axC.spines['left'].set_visible(False)
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    axC.xaxis.set_inverted(True)
    axC.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])

    # save & show
    if savename_mode:
        #plt.tight_layout()
        savepath = '../Figures/0_paper/fig_4_multiple_trajectory_examples.pdf'
        #fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=1800, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)
    
    
    
# Fig. 5
def fig_etiology_tuning_curve_broadening(mEPSC_avg_norm_CTR, mEPSC_avg_norm_FR, lognormal_function, tuning_curves_ms_CTR, tuning_curves_ms_FR, tuning_curves_nms_CTR, tuning_curves_nms_FR, results_membrane_noise_CTR, results_membrane_noise_FR, colors=['black', 'red', 'steelblue'], fontsizes={'panelletterfontsize': 15}, figsize=(7,5), savename_mode=False):
    # compose figure showing etiology of tuning-curve broadening
    
    # input
    # mEPSC_avg_norm_CTR/FR are arrays of experimental mEPSC distributions
    # lognormal_function is the fitting function for the mEPSC distributions
    # tuning_curves_exp_CTR/FR are experimental tuning curves
    # tuning_curves_ms_CTR/FR are multiplicative-scaling tuning curves
    # tuning_curves_nms_CTR/FR are non-multiplicative scaling tuning curves
    # results_membrane_noise_CTR/FR are dictionaries containing experimental membrane noise data
    # titles is a list of 
    # colors are the colors for color_CTR, color_FR, color_CTR_mult_scale
    # fontsizes is a dictionary of used font sizes
    # savename_mode decides whether the figure is saved or not
        
    color_CTR, color_FR, color_FR_non_mult_scale = colors[0], colors[1], colors[2]
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3,4,figure=fig,width_ratios=[1,1,1,1],height_ratios=[1,0.05,1],wspace=0.8,hspace=0.4)

    # panel A
    axA = fig.add_subplot(gs[0,0:2])
    pf.plot_mEPSC_lognormal_fit(mEPSC_avg_norm_CTR, mEPSC_avg_norm_FR, lognormal_function=pf.lognormal, label_mode="short", scale_factor=1.845, bins=500, color_CTR=color_CTR, color_FR=color_FR_non_mult_scale, color_scaled=color_FR, title=None, ax=axA) # swap colors for consistency --> multiplicative scaling should be red
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])

    # panels B–D
    #panels = [fig.add_subplot(gs[0,2]), fig.add_subplot(gs[0,3]), fig.add_subplot(gs[0,4])]
    #datasets = [
    #    (tuning_curves_exp_CTR, tuning_curves_exp_FR, 'Experiment \n'),
    #    (tuning_curves_ms_CTR,  tuning_curves_ms_FR,  'Simulation \n (mult.)'),
    #    (tuning_curves_nms_CTR, tuning_curves_nms_FR, 'Simulation \n (non-mult.)')]
    #letters = ['B','C','D']

    #for ax, (ctr, fr, title), L in zip(panels, datasets, letters):
    #    pf.plot_tuning_curves(ctr, fr, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR, show_legend=False, ax=ax)
    #    ax.set_title(title)
    #    ax.set_xlabel('Distance from \n Preferred ($\circ$)')
    #    ppf.panel_letter(ax, L)
    #    if L != 'B':
    #        ax.set_ylabel('')
    #        ax.set_yticklabels([])

    # B: Simulation (mult. scaling)
    axB = fig.add_subplot(gs[0, 2])
    pf.plot_tuning_curves(tuning_curves_ms_CTR, tuning_curves_ms_FR, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR, show_legend=False, show_xlabel=True, show_ylabel=True, ax=axB)
    axB.set_title('AdExp \n (mult.)') # scaled
    axB.set_xlabel('Distance from \n Pref. Orientation ($\circ$)')
    #axB.set_yticklabels([])
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    # C: Simulation (non-mult. scaling)
    axC = fig.add_subplot(gs[0, 3])
    pf.plot_tuning_curves(tuning_curves_nms_CTR, tuning_curves_nms_FR, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR_non_mult_scale, show_legend=False, show_xlabel=True, show_ylabel=False, ax=axC)
    axC.set_title('AdExp \n (non-mult.)') # scaled
    axC.set_xlabel('Distance from \n Pref. Orientation ($\circ$)')
    axC.set_yticklabels([])
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    
    # panel D (sigma = 0)
    axD = fig.add_subplot(gs[2,0])
    noise_index_0 = 0
    tcs_CTR_0 = [np.array(tc[noise_index_0]) for tc in results_membrane_noise_CTR['tuning_curve']]
    tcs_FR_0  = [np.array(tc[noise_index_0]) for tc in results_membrane_noise_FR['tuning_curve']]
    pf.plot_tuning_curves(tcs_CTR_0, tcs_FR_0, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR, half_width_max=True, minmax_mode=False, show_legend=False, ax=axD)
    axD.set_title(r'$\sigma=0$ mV/ms')
    axD.set_xlabel('Distance from \n Pref. Orientation ($\circ$)')
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    # panel E (OSI vs noise)
    axE = fig.add_subplot(gs[2,1:3])
    pf.plot_membrane_noise_effect(results_membrane_noise_CTR, results_membrane_noise_FR, 'OSI', error_bars=False, half_width_max=True, colors=[color_CTR, color_FR], ax=axE)
    axE.set_title('OSI and FWHM for different membrane noise levels')
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'], dx=-0.07, dy=0.02)

    # panel F (sigma = 8)
    axF = fig.add_subplot(gs[2,3])
    noise_index_8 = 5
    tcs_CTR_8 = [np.array(tc[noise_index_8]) for tc in results_membrane_noise_CTR['tuning_curve']]
    tcs_FR_8  = [np.array(tc[noise_index_8]) for tc in results_membrane_noise_FR['tuning_curve']]
    pf.plot_tuning_curves(tcs_CTR_8, tcs_FR_8, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR, half_width_max=True, minmax_mode=False, show_legend=False, ax=axF)
    axF.set_title(r'$\sigma=4$ mV/ms')
    axF.set_ylabel('')
    axF.set_xlabel('Distance from \n Pref. Orientation ($\circ$)')
    #axF.set_yticklabels([])
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])

    if savename_mode is True:
        #plt.tight_layout()
        savepath='../Figures/0_paper/fig_5_etiology_of_tc_broadening.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        plt.show()

    plt.show()
    


# Fig. 6
def fig_V_gap(results_multiple_CTR_FR_runs_V_gap_AdExp, results_V_gap_variable_E_L, results_V_gap_variable_V_thresh, colors=['black', 'red', 'darkcyan', 'darkviolet'], fontsizes={'panelletterfontsize': 15}, figsize=(7,5), savename_mode=False):
    # compose figure about the etiology of tuning curve
    
    # input
    # results_multiple_CTR_FR_runs_V_gap_AdExp is the dictionary of V_gap_constant results
    # results_V_gap_variable_E_L, results_V_gap_variable_V_thresh are the dictionary of V_gap_variable
    # colors are the colors for color_CTR, color_FR, color_V_rest_var, color_V_thresh_var
    # fontsizes is a dictionary of used font sizes
    # savename_mode determines if the figures is saved or not
    
    color_CTR, color_FR, color_V_rest_var, color_V_thresh_var = colors[0], colors[1], colors[2], colors[3]
    
    fig = plt.figure(figsize=figsize)
    # 3 rows x 5 columns; keep last column mostly unused to allow a clean 2-col span for OSI
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1,1], height_ratios=[0.8,1,1], wspace=0.2, hspace=0.2)
    
    # create spiek shape for illustrative figures
    t_plot, V_plot = pf.illustrative_spike()

    # A: 
    gsAIl = gs[0,0].subgridspec(1, 2, width_ratios=[1,1], wspace=0.05)
    axAIl_plot = fig.add_subplot(gsAIl[0,0])
    axAIl_text = fig.add_subplot(gsAIl[0,1])
    
    pf.plot_illustrative_sliding_variable_gap(t_plot, V_plot, color_1 = color_V_rest_var, color_2 = color_V_thresh_var, ax=axAIl_plot) #, fontsize=9
    #axAIl.text(1, 1, "Sliding variable \nvoltage gap", transform=axAIl.transAxes, va="center", ha="left", fontsize=fontsizes.get('title', 16))
    axAIl_text.axis("off")
    axAIl_text.text(0.05, 0.5, "Sliding variable\nvoltage gap", transform=axAIl_text.transAxes, va="center", ha="left", fontsize=fontsizes.get('title', 11))
    #panel_letter(axAIl_plot, "A", size=fontsizes['panelletterfontsize'])
    axAIl_plot.text(-0.12, 0.8, "A", transform=axAIl_plot.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    axAT = fig.add_subplot(gs[1,0])
    pf.plot_V_gap_variable(results_V_gap_variable_E_L, results_V_gap_variable_V_thresh, 'OSI', mean_over_zeros=True, description_mode=False, colors=[color_V_rest_var, color_V_thresh_var], ax=axAT, savename_mode=False)
    #axAT.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    #axAT.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    axAT.get_legend().remove()
    axAT.set_xticklabels([])
    axAT.set_xlabel("")
    #axAT.set_title("Sliding variable voltage gap")
    
    
    axAB = fig.add_subplot(gs[2,0])
    pf.plot_V_gap_variable(results_V_gap_variable_E_L, results_V_gap_variable_V_thresh, 'E_tot', mean_over_zeros=True, description_mode=False, colors=[color_V_rest_var, color_V_thresh_var], ax=axAB, savename_mode=False)
    #axAB.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    #axAB.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    axAB.set_ylabel("$E_{tot}$ ($10^{9}$ ATP/s)")
    axAB.set_xlabel("$V_{thresh} - V_{rest}$ (mV)")
    axAB.legend(frameon=False)
    
    gsBIl = gs[0,1].subgridspec(1, 2, width_ratios=[1,1], wspace=0.05)
    axBIl_plot = fig.add_subplot(gsBIl[0,0])
    axBIl_text = fig.add_subplot(gsBIl[0,1])
    axBIl_text.axis("off")
    pf.plot_illustrative_sliding_constant_gap(t_plot, V_plot, color_1=color_CTR, color_2=color_FR, ax=axBIl_plot) # , fontsize=9
    #axBIl.text(1, 1, "Sliding constant \nvoltage gap", transform=axBIl.transAxes, va="center", ha="left", fontsize=fontsizes.get('title', 16))
    axBIl_text.text(0.05, 0.5, "Sliding constant\nvoltage gap", transform=axBIl_text.transAxes, va="center", ha="left", fontsize=fontsizes.get('title', 11))
    #panel_letter(axBIl_plot, "B", size=fontsizes['panelletterfontsize'])
    axBIl_plot.text(-0.12, 0.8, "B", transform=axBIl_plot.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    axBT = fig.add_subplot(gs[1,1])
    pf.plot_CTR_FR_V_gap_constant(results_multiple_CTR_FR_runs_V_gap_AdExp, ['OSI'], CTR_FR=True, shade_intersection=True, error_bars=False, mean_over_zeros=True, description_mode=False, colors=[color_CTR, color_FR], ax=axBT, savename_mode=True)
    #axBT.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    #axBT.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    axBT.get_legend().remove()
    axBT.set_xticklabels([])
    axBT.set_xlabel("")
    #axBT.set_ylabel("")
    #axBT.set_title("Sliding constant voltage gap")
    
    
    axBB = fig.add_subplot(gs[2,1])
    pf.plot_CTR_FR_V_gap_constant(results_multiple_CTR_FR_runs_V_gap_AdExp, ['E_tot'], CTR_FR=True, shade_intersection=True, error_bars=False, mean_over_zeros=True, description_mode=False, colors=[color_CTR, color_FR], ax=axBB, savename_mode=True)
    #axBB.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    #axBB.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    axBB.set_ylabel("$E_{tot}$ ($10^{9}$ ATP/s)")
    axBB.set_xlabel("$V_{rest}$ (mV)")
    axBB.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, loc='upper right', bbox_to_anchor=(1.02, 1.05)) #
    #axBB.set_ylabel("")

    if savename_mode is True:
        #plt.tight_layout()
        savepath='../Figures/0_paper/fig_6_V_gap.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        plt.show()

    plt.show()


# Fig. S1
def fig_synaptic_input(w_e_0, r_e, N_e_noise, spike_times_e, spike_times_i, w_e_CTR, w_e_FR, colors=['#2e7d32', '#f9a825', '#f9a825', 'lightskyblue', 'black', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(5,7), savename_mode=True):
    # create figure on synaptic input structure and stimulation protocol
    # input
    # w_e_0 is an array of length N_e with normalized excitatory synaptic weights
    # r_e is a list of arrays with firing rates for excitatory neurons in Hz
    # N_e_noise is the number excitatory noise synapses
    # spike_times_e, spike_times_i is a dictionary or list of spike times for excitatory & inhibitory neurons
    # w_e_CTR, w_e_FR are arrays of length N_e with scaled excitatory synaptic weights for CTR & FR case
    # colors are the colors for color_w_e, color_r_e, color_input_exc, color_input_inh, color_CTR, color_FR
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    color_w_e, color_r_e, color_input_exc, color_input_inh, color_CTR, color_FR = colors[0], colors[1], colors[2], colors[3], colors[4], colors[5]
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 5, figure=fig, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], height_ratios=[1, 2, 0.4, 2], wspace=1.0, hspace=0.15)  

    w_e = pf.w_scale_to_w_e_syn(w_e_0, len(w_e_0))
    w_e = w_e / np.mean(w_e)
    
    # A: stimulation protocol image
    axA = fig.add_subplot(gs[0, :])
    imgA = plt.imread('../Figures/0_paper/matplotlib_figs/stimulation_protocol.png')
    axA.imshow(imgA)
    axA.axis('off')
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    #axA.text(-0.02, 1.05, 'A', transform=axA.transAxes, fontweight='bold', va='bottom', ha='left')

    # B: excitatory synaptic weights & firing rates distribution
    gsBC = gs[1, :].subgridspec(1, 2, width_ratios=[1, 3], wspace=0.3)
    gsB = gsBC[0, 0].subgridspec(2, 1, hspace=0.6)
    gsC = gsBC[0, 1].subgridspec(2, 2, wspace=0.3, hspace=0.6)

    axB_w = fig.add_subplot(gsB[0, 0])
    axB_r = fig.add_subplot(gsB[1, 0])

    # get signalling parts
    w_e_signal = w_e[N_e_noise:]
    r_e_signal = r_e[N_e_noise:]
    
    # get rates of respective stimulus presentation
    r_e_signal_0   = r_e_signal[:, 3000 + 750]  
    r_e_signal_30  = r_e_signal[:, 6000 + 750]  
    r_e_signal_60  = r_e_signal[:, 9000 + 750]  
    r_e_signal_90  = r_e_signal[:, 12000 + 750] 

    pf.plot_one_histogram(w_e, 'exc weights (nS)', '$n_{syn}$', '$w_{e}$', description=None, color=color_w_e, axis_mode='log', ax=axB_w)
    axB_w.get_legend().remove() if axB_w.get_legend() is not None else None
    pf.plot_one_histogram(np.concatenate([r_e_signal_0, r_e_signal_0]), 'exc rates (Hz)', '$n_{syn}$', '$r_{e}$', description=None, color=color_r_e, axis_mode='linear', ax=axB_r)
    axB_r.get_legend().remove() if axB_r.get_legend() is not None else None
    panel_letter(axB_w, "B", size=fontsizes['panelletterfontsize'])
    
    # C: excitatory synaptic weights & firing rates matching logic for different stimuli
    axC_0  = fig.add_subplot(gsC[0, 0])
    axC_30 = fig.add_subplot(gsC[0, 1])
    axC_60 = fig.add_subplot(gsC[1, 0])
    axC_90 = fig.add_subplot(gsC[1, 1])

    #w_e = pf.w_scale_to_w_e_syn(w_e_signal, len(w_e_0))*np.mean(w_e_signal)
    pf.plot_synapse_weights_and_rates(w_e_signal, r_e_signal_0, 'Synaptic input 0°', ax_bottom_mode=False, ax_left_mode=True, ax_right_mode=False, colors=[color_w_e, color_r_e], ax=axC_0) # inset_img_path=icon(0),  inset_pos=[0.56, 0.55, 0.4, 0.4]
    pf.plot_synapse_weights_and_rates(w_e_signal, r_e_signal_30, 'Synaptic input 30°', ax_bottom_mode=False, ax_left_mode=False, ax_right_mode=True, colors=[color_w_e, color_r_e], ax=axC_30) # inset_img_path=icon(30),  inset_pos=[0.56, 0.55, 0.4, 0.4]
    pf.plot_synapse_weights_and_rates(w_e_signal, r_e_signal_60, 'Synaptic input 60°', ax_bottom_mode=True, ax_left_mode=True, ax_right_mode=False, colors=[color_w_e, color_r_e], ax=axC_60) # inset_img_path=icon(60),  inset_pos=[0.04, 0.55, 0.4, 0.4]
    pf.plot_synapse_weights_and_rates(w_e_signal, r_e_signal_90, 'Synaptic input 90°', ax_bottom_mode=True, ax_left_mode=False, ax_right_mode=True, colors=[color_w_e, color_r_e], ax=axC_90) # inset_img_path=icon(90),  inset_pos=[0.04, 0.55, 0.4, 0.4]
    panel_letter(axC_0, "C", size=fontsizes['panelletterfontsize'])

    # add small inset fig
    base_img_path = '../Figures/0_paper/matplotlib_figs'
    icon = lambda deg: os.path.join(base_img_path, f'{deg}degree.png')

    ax_list = [axC_0, axC_30, axC_60, axC_90]
    deg_list = (0, 30, 60, 90)
    pos_list = [[0.45, 0.55, 0.6, 0.6], [0.45, 0.55, 0.6, 0.6], [-0.13, 0.55, 0.6, 0.6], [-0.13, 0.55, 0.6, 0.6]]
   
    for ax, deg, pos in zip(ax_list, deg_list, pos_list): 
        inset_img_path = icon(deg)
        if inset_img_path and os.path.exists(inset_img_path):
            inset = ax.inset_axes(pos)
            inset.imshow(plt.imread(inset_img_path))
            inset.axis('off')

    # D: raster plot
    axD = fig.add_subplot(gs[3, 0:3])

    # get text writer function here!
    pf.plot_raster(spike_times_e, spike_times_i, N_e_noise, w_e=w_e, title_mode=False, colors=[color_input_exc, color_input_inh], ax=axD)
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    # E: plot cumulative weight functions
    gsE = gs[3, 3:5].subgridspec(2, 1, hspace=0.5)
    axE_top = fig.add_subplot(gsE[0, 0])
    axE_bottom = fig.add_subplot(gsE[1, 0])

    pf.plot_cumulative_weights(w_e_CTR, 'CTR', color=color_CTR, ax=axE_top)
    axE_top.set_xlabel('')
    pf.plot_cumulative_weights(w_e_FR, 'FR', color=color_FR, ax=axE_bottom)
    panel_letter(axE_top, "E", size=fontsizes['panelletterfontsize'])
    
    #plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S1_synaptic_input.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    
    plt.show()


# Fig. S2
def fig_3x3_tuning_curves_N_exc_signal(model_mode, N_e_signal_ratios, savename_basic, normalized=True, mean_over_zeros=False, mode='tuning_curve', colors=['black', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(5,6), savename_mode=False):
    # plot a 3x3 grid of CTR vs FR tuning curves across N_e_signal_ratio values
    
    # input
    # model_mode is a string ('LIF' or 'AdExp') deciding which model’s results to load
    # N_e_signal_ratios is a list of 9 ratios (e.g., [0.1, ..., 0.9]) for the subplot titles & file names
    # savename_basic is the base of the saved file names (e.g., 'tuning_curve_N_e_signal_ratio')
    # normalized determines whether to normalize tuning curves to their maxima
    # mean_over_zeros determines whether to exclude all-zero runs before averaging
    # mode determines what to plot 'tuning_curve' or 'CV_ISI_tuning_curve' for y-axis wording
    # colors are the colors for color_CTR, color_FR
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    color_CTR, color_FR = colors[0], colors[1]
    
    letters=("A","B","C","D","E","F","G","H","I")
    
    # figure & layout
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 3, figure=fig, wspace=0.2, hspace=0.4)
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(9)]

    # helper to load one ratio’s data and return CTR/FR tuning curves
    def load_tc(percent):
        base = f"{savename_basic}_{percent}"
        name_CTR = f"{base}_results_single_runs_CTR_{model_mode}"
        name_FR  = f"{base}_results_single_runs_FR_{model_mode}"
        results_CTR  = af.load_data(name_CTR)  # expects a dict with key "tuning_curve"
        results_FR   = af.load_data(name_FR)
        return results_CTR["tuning_curve"], results_FR["tuning_curve"]

    # populate subplots
    for k, ratio in enumerate(N_e_signal_ratios):
        ax = axes[k]
        percent = int(round(ratio * 100))
        tuning_curves_CTR, tuning_curves_FR = load_tc(percent)

        # show legend only on the first (top-left panel 
        show_legend = (k == 0) #show_legend = (k == 8) last (bottom-right) panel

        # show y-labels only on the first column (k % 3 == 0)
        show_ylabel = (k % 3 == 0)

        # show x-labels only on the bottom row (k // 3 == 2)
        show_xlabel = (k // 3 == 2)

        pf.plot_tuning_curves(tuning_curves_CTR, tuning_curves_FR, label_CTR='CTR', label_FR='FR', normalized=normalized, color_CTR = color_CTR, color_FR = color_FR, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=show_legend, show_xlabel=show_xlabel, show_ylabel=show_ylabel, ax=ax, savename=None)

        # remove x-ticks if no xlabel
        if not show_xlabel:
            ax.set_xticklabels([])

        # remove y-ticks if no ylabel
        if not show_ylabel:
            ax.set_yticklabels([])
            
        # title with the ratio (e.g., "0.1", "0.2", ... "0.9")
        ax.set_title(f"{percent:.0f}% exc sig syn")

        # panel letter (top-left corner)
        #ax.text(-0.12, 1.03, letters[k], transform=ax.transAxes, fontweight='bold', va='bottom', ha='left') #, fontsize=20
        panel_letter(ax, letters[k], size=fontsizes['panelletterfontsize'])

    #fig.suptitle(f"{model_mode} — fraction of excitatory signaling synapses", y=1.00002)

    
    if savename_mode is True:
        #plt.tight_layout()
        savepath='../Figures/0_paper/fig_S2_N_exc_signal_grid_AdExp.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        #print(f"saved: {savepath}")
        plt.show()

    plt.show()

# Fig. S3   
def fig_tuning_and_phaseplanes_2x2(results_single_runs_CTR_LIF, results_single_runs_FR_LIF, results_single_runs_CTR_LIFad, results_single_runs_FR_LIFad, results_single_runs_CTR_LIFexp, results_single_runs_FR_LIFexp, results_single_runs_CTR_AdExp, results_single_runs_FR_AdExp, g_L_CTR, g_L_FR, E_L_CTR, E_L_FR, V_thresh, Delta_T_ad, a_ad, tau_w_ad, normalized=True, mean_over_zeros=False, mode='tuning_curve', titles=("LIF","LIF+Ad","LIF+Exp","AdExp"), colors=['black', 'red'], fontsizes={'panelletterfontsize':15}, figsize=(9,4), savename_mode=None):
    # paper-ready 2x2 figure combining tuning curves and phase planes
    
    color_CTR, color_FR = colors[0], colors[1]
    
    # extract tuning curves directly from results
    tuning_curves_CTR_LIF    = results_single_runs_CTR_LIF['tuning_curve']
    tuning_curves_FR_LIF     = results_single_runs_FR_LIF['tuning_curve']
    tuning_curves_CTR_LIFad  = results_single_runs_CTR_LIFad['tuning_curve']
    tuning_curves_FR_LIFad   = results_single_runs_FR_LIFad['tuning_curve']
    tuning_curves_CTR_LIFexp = results_single_runs_CTR_LIFexp['tuning_curve']
    tuning_curves_FR_LIFexp  = results_single_runs_FR_LIFexp['tuning_curve']
    tuning_curves_CTR_AdExp  = results_single_runs_CTR_AdExp['tuning_curve']
    tuning_curves_FR_AdExp   = results_single_runs_FR_AdExp['tuning_curve']

    # mean synaptic currents (modify to include I_syn_i if needed)
    I_syn_mean_CTR_LIF    = np.mean(results_single_runs_CTR_LIF['I_syn_e'])
    I_syn_mean_FR_LIF     = np.mean(results_single_runs_FR_LIF['I_syn_e'])
    I_syn_mean_CTR_LIFad  = np.mean(results_single_runs_CTR_LIFad['I_syn_e'])
    I_syn_mean_FR_LIFad   = np.mean(results_single_runs_FR_LIFad['I_syn_e'])
    I_syn_mean_CTR_LIFexp = np.mean(results_single_runs_CTR_LIFexp['I_syn_e'])
    I_syn_mean_FR_LIFexp  = np.mean(results_single_runs_FR_LIFexp['I_syn_e'])
    I_syn_mean_CTR_AdExp  = np.mean(results_single_runs_CTR_AdExp['I_syn_e'])
    I_syn_mean_FR_AdExp   = np.mean(results_single_runs_FR_AdExp['I_syn_e'])

    # figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, wspace=0.4, hspace=0.4, width_ratios=[1,2,1,2]) # , hspace=0.6

    # axes (2 rows, 4 columns: tuning | phase)
    axA_tune  = fig.add_subplot(gs[0,0])
    axA_phase = fig.add_subplot(gs[0,1])
    axB_tune  = fig.add_subplot(gs[0,2])
    axB_phase = fig.add_subplot(gs[0,3])
    axC_tune  = fig.add_subplot(gs[1,0])
    axC_phase = fig.add_subplot(gs[1,1])
    axD_tune  = fig.add_subplot(gs[1,2])
    axD_phase = fig.add_subplot(gs[1,3])

    # A: LIF
    pf.plot_tuning_curves(tuning_curves_CTR_LIF, tuning_curves_FR_LIF, label_CTR='CTR', label_FR='FR', color_CTR=color_CTR, color_FR=color_FR, normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=False, show_xlabel=True, show_ylabel=True, ax=axA_tune)
    axA_tune.set_title(titles[0])
    panel_letter(axA_tune, "A", size=fontsizes['panelletterfontsize'])
    axA_tune.set_xlabel("")
    axA_tune.set_xticklabels([])
    pf.plot_phase_plane_Vw(I_syn_mean_CTR_LIF, I_syn_mean_FR_LIF, g_L_CTR, g_L_FR, E_L_CTR, E_L_FR, V_thresh, 0.0, 0.0, tau_w_ad, 'LIF', -70.0, V_thresh+1.0, 400, colors=[color_CTR, color_FR], ax=axA_phase)
    axA_phase.set_xlabel("")
    axA_phase.set_xticklabels([])
    axA_phase.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', loc='center left', bbox_to_anchor=(0.0, 0.25))
    

    # B: LIF + Ad 
    pf.plot_tuning_curves(tuning_curves_CTR_LIFad, tuning_curves_FR_LIFad, label_CTR='CTR', label_FR='FR', color_CTR=color_CTR, color_FR=color_FR, normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=False, show_xlabel=True, show_ylabel=True, ax=axB_tune)
    axB_tune.set_title(titles[1])
    axB_tune.set_xlabel("")
    axB_tune.set_xticklabels([])
    panel_letter(axB_tune, "B", size=fontsizes['panelletterfontsize'])
    pf.plot_phase_plane_Vw(I_syn_mean_CTR_LIFad, I_syn_mean_FR_LIFad, g_L_CTR, g_L_FR, E_L_CTR, E_L_FR, V_thresh, 0.0, a_ad, tau_w_ad, 'LIF+Ad', -70.0, V_thresh+1.0, 400, colors=[color_CTR, color_FR], ax=axB_phase)
    axB_phase.set_xlabel("")
    axB_phase.set_xticklabels([])
    # only plot w-nullclines legend
    handles, labels = axB_phase.get_legend_handles_labels()
    new_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'V-nullcline CTR' not in l and 'V-nullcline FR' not in l]
    new_handles  = [h for h, l in new_handles_labels] # keep only w-nullclines
    new_labels   = [l for h, l in new_handles_labels] # keep only w-nullclines
    axB_phase.legend(new_handles, new_labels, frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', loc='center left', bbox_to_anchor=(0.0, 0.35))

    # C: LIF + Exp
    pf.plot_tuning_curves(tuning_curves_CTR_LIFexp, tuning_curves_FR_LIFexp, label_CTR='CTR', label_FR='FR', color_CTR=color_CTR, color_FR=color_FR, normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=False, show_xlabel=True, show_ylabel=True, ax=axC_tune)
    axC_tune.set_title(titles[2])
    panel_letter(axC_tune, "C", size=fontsizes['panelletterfontsize'])
    pf.plot_phase_plane_Vw(I_syn_mean_CTR_LIFexp, I_syn_mean_FR_LIFexp, g_L_CTR, g_L_FR, E_L_CTR, E_L_FR, V_thresh, Delta_T_ad, 0.0, tau_w_ad, 'LIF+Exp', -70.0, V_thresh+1.0, 400, colors=[color_CTR, color_FR], ax=axC_phase)
    axC_phase.get_legend().remove()

    # D: AdExp
    pf.plot_tuning_curves(tuning_curves_CTR_AdExp, tuning_curves_FR_AdExp, label_CTR='CTR', label_FR='FR', color_CTR=color_CTR, color_FR=color_FR, normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=True, show_xlabel=True, show_ylabel=True, ax=axD_tune)
    axD_tune.set_title(titles[3])
    #axD_tune.set_yticklabels([])
    panel_letter(axD_tune, "D", size=fontsizes['panelletterfontsize'])
    pf.plot_phase_plane_Vw(I_syn_mean_CTR_AdExp, I_syn_mean_FR_AdExp, g_L_CTR, g_L_FR, E_L_CTR, E_L_FR, V_thresh, Delta_T_ad, a_ad, tau_w_ad, 'AdExp', -70.0, V_thresh+1.0, 400, colors=[color_CTR, color_FR], ax=axD_phase)
    axD_phase.get_legend().remove()

    plt.tight_layout()

    if savename_mode is True:
        #plt.tight_layout()
        savepath='../Figures/0_paper/fig_S3_ad_or_exp_tuning_and_phase.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        #print(f"saved: {savepath}")        

    plt.show()

        
# Fig. S4
def fig_information(results_mean_spiking_trials, fontsizes={'panelletterfontsize': 15}, figsize=(7,6), savename_mode=True): #, figsize=(13, 11)
    # create information figure
    # input
    # results_mean_spiking_trials is a dictionary of the full grid results
    # results_grid_point_to_exp_data is a dictionary of the experimental data points fitted to their closest grid point 
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    # load data
    R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy, MI_new, MI_new_per_energy, MICE_new, MICE_new_per_energy, TE_new, TE_new_per_energy, TECE_new, TECE_new_per_energy = pf.load_data_correlations(results_mean_spiking_trials)

    # define figure
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3,2, figure=fig, wspace=0.3, hspace=0.4)
    
    axA = fig.add_subplot(gs[0,0])
    pf.plot_correlation(r_post, CV_V_m, 'r_post', 'CV_V_m', z = E_tot, z_label='E_tot', show_colorbar=False, plot_mode='correlation_short', ax=axA)
    #axA.set_xlim(right=25) 
    axA.set_xticklabels([])
    axA.set_xlabel("")
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])

    axB = fig.add_subplot(gs[0,1])
    pf.plot_correlation(r_post, CV_ISI, 'r_post', 'CV_ISI', z = E_tot, z_label='E_tot', show_colorbar=True, plot_mode='correlation_short', ax=axB)
    #axB.set_xlim(right=25) 
    axB.set_xticklabels([])
    axB.set_xlabel("")
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    axC = fig.add_subplot(gs[1,0])
    pf.plot_correlation(r_post, MI, 'r_post', 'MI', z = E_tot, z_label='E_tot', show_colorbar=False, plot_mode='correlation_short', ax=axC)
    #axC.set_xlim(right=25) 
    axC.set_xticklabels([])
    axC.set_xlabel("")
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])

    axD = fig.add_subplot(gs[1,1])
    pf.plot_correlation(r_post, TE, 'r_post', 'TE', z = E_tot, z_label='E_tot', show_colorbar=True, plot_mode='correlation_short', ax=axD)
    #axD.set_xlim(right=25) 
    axD.set_xticklabels([])
    axD.set_xlabel("")
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    axE = fig.add_subplot(gs[2,0])
    pf.plot_correlation(r_post, MI_tuning_curve, 'r_post', 'MI_tuning_curve', show_colorbar=False, z = E_tot, z_label='E_tot', plot_mode='correlation_short', ax=axE)
    #axE.set_xlim(right=25) 
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'])

    axF = fig.add_subplot(gs[2,1])
    pf.plot_correlation(r_post, OSI, 'r_post', 'OSI', z = E_tot, z_label='E_tot', show_colorbar=True, plot_mode='correlation_short', ax=axF) # , results_grid_point_to_exp_data=results_grid_point_to_exp_data
    #axF.set_xlim(right=25) 
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])
    
    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S4_information.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)

# Fig. S5
def fig_info_binsizes(binning_results, fontsizes={'panelletterfontsize': 15}, figsize=(7,6), savename_mode=True):
    # create information figure
    # input
    # binning_results is the results of the binning process
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not 
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig, wspace=0.5)
    
    optimal_MI = (1200, 150)
    #optimal_TE = (500, 58) #OLD: actually 294, but does not plot correctly
    optimal_TE = (350,108) #actually (782, 39)
    
    # unique bin values from binning_results
    br = np.asarray(binning_results)
    bin_t_vals = np.unique(br[:, 0])
    bin_r_vals = np.unique(br[:, 1])
    
    def bin_to_index(opt):
        # map desired bin sizes to the indices of the closest available values
        t, r = opt
    
        # index of closest time bin
        x_idx = int(np.argmin(np.abs(bin_t_vals - t)))
        # index of closest rate bin
        y_idx = int(np.argmin(np.abs(bin_r_vals - r)))
        return x_idx, y_idx

    x_MI, y_MI = bin_to_index(optimal_MI)
    x_TE, y_TE = bin_to_index(optimal_TE)
    
    # plot experimental data
    axA = fig.add_subplot(gs[0, 0])
    pf.plot_bin_sizes(binning_results, 'MI', ax=axA)
    axA.scatter(x_MI, y_MI, s=200, facecolor="white", linewidth=1.5, zorder=10)
    axA.text(x_MI, y_MI, "A", ha="center", va="center", fontsize=fontsizes['panelletterfontsize']*0.8, zorder=11)
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    axB = fig.add_subplot(gs[0, 1])
    pf.plot_bin_sizes(binning_results, 'TE', ax=axB)
    axB.scatter(x_TE, y_TE, s=200, facecolor="white", linewidth=1.5, zorder=10)
    axB.text(x_TE, y_TE, "B", ha="center", va="center", fontsize=fontsizes['panelletterfontsize']*0.8, zorder=11)
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    #plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S5_info_binsizes.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    
    

        
# Fig. S6 
def fig_A_syn_estimation(currents_subthreshold, colors=['#57e7ff', '#9357ff', '#ff1d1d'], fontsizes={'panelletterfontsize': 15}, figsize=(7,2), savename_mode=True): 
    # create A_syn estimation figure
    # input
    # currents_subthreshold is a dictionary of ionic sub-threshold fluctuation currents in nA 
    # colors are the colors for color_K, color_Na, color_Ca
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1,2,figure=fig,width_ratios=[1.0, 1.0],wspace=0.1)

    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[0,1])
    
    # A
    # ionic currents during subthreshold fluctuations
    t_range_st = currents_subthreshold["t_vec"]
    I_Na_st = currents_subthreshold["I_Na"]
    I_Ca_st = currents_subthreshold["I_Ca"]
    I_K_st = currents_subthreshold["I_K"]
    
    label_K, label_Na, label_Ca = '$I_{K^{+}}$', '$I_{Na^{+}}$', '$I_{Ca^{2+}}$'
    color_K, color_Na, color_Ca = colors[0], colors[1], colors[2] #'#57e7ff', '#9357ff', '#ff1d1d'
    x_label, y_label = 'time (ms)', 'ionic current (nA)'
    pf.plot_ionic_current(t_range_st, I_K_st, I_Na_st, I_Ca_st, label_K, label_Na, label_Ca, color_K, color_Na, color_Ca, x_label, y_label, None, ax=axA)
    #axA.get_legend().remove()
    axA.legend(facecolor='white', edgecolor='none', framealpha=1.0)
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    
    # B
    img_path = "../Figures/0_paper/matplotlib_figs/l23r.png"
    img = plt.imread(img_path)
    img_rot = np.rot90(img)
    
    axB.imshow(img_rot)
    axB.axis("off")  # remove axes around the image   
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])
    
    
    #plt.tight_layout()
    if savename_mode is True:
        mpl.rcParams['pdf.compression'] = 0
        savepath="../Figures/0_paper/fig_S6_A_syn_estimation.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=False)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    
    plt.show()
    

# Fig. S7  
def fig_adaptation_currents(r_post_list, w_ad_list, ratio_list, labels_list, AP_adaptation, ionic_currents_adaptation, colors=['blue', 'orange', '#57e7ff', '#9357ff', '#ff1d1d'], fontsizes={'panelletterfontsize': 15}, figsize=(7,6), savename_mode=True): 
    # create adaptation currents figure
    # input
    # r_post_list is a list of firing rates in Hz
    # w_ad_list is a list of adaptation currents in nA (for different adaptation time constants tau_w_ad)
    # ratio_list is a list of ratios between adaptation currents and excitatory synaptic input currents (for different adaptation time constants tau_w_ad)
    # labels_list is a list of different adaptation time constants tau_w_ad labels
    # AP_adaptation is a dictionary of the membrane voltage during adaptation
    # ionic_currents_adaptation is a dictionary of the ionic currents during adaptation
    # colors are the colors for colors_ad, color_AP_no_ad, color_AP_ad, color_K, color_Na, color_Ca 
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    colors_ad, color_AP_no_ad, color_AP_ad, color_K, color_Na, color_Ca = colors[0], colors[1], colors[2], colors[3], colors[4], colors[5]  #'#57e7ff', '#9357ff', '#ff1d1d'
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2,2,figure=fig,width_ratios=[1.0,1.0],height_ratios=[1.0,1.0],wspace=0.3,hspace=0.5)

    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[0,1])
    axC = fig.add_subplot(gs[1,0])
    axD = fig.add_subplot(gs[1,1])
    
    # A
    pf.plot_template_n_graph(r_post_list, w_ad_list, labels_list, x_label='$r_{post}$ (Hz)', y_label='$I_{ad}$ (nA)', title=None, colors=colors_ad, yerr_list=None, ax=axA)
    axA.get_legend().remove()
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    
    # B
    pf.plot_template_n_graph(r_post_list, ratio_list, labels_list, x_label='$r_{post}$ (Hz)', y_label='$I_{ad}$ / $I_{syn,e}$ (%)', title=None, colors=colors_ad, yerr_list=None, ax=axB)
    axB.legend(frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', reverse=True)
    #axB.legend(*axB.get_legend_handles_labels()[::-1], frameon=True, facecolor='white', framealpha=1.0, edgecolor='none')
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    # C
    t_vec_1 = AP_adaptation["t_vec_1"]
    t_vec_2 = AP_adaptation["t_vec_2"]
    V_m_adaptation = AP_adaptation["V_m_adaptation"]
    V_m_no_adaptation = AP_adaptation["V_m_no_adaptation"]

    pf.plot_template_two_graph(t_vec_1, V_m_adaptation, t_vec_2, V_m_no_adaptation, '', '', 'adaptation', 'no adaptation', None, color_1=color_AP_ad, color_2=color_AP_no_ad, ax=axC)
    x0 = t_vec_1[0]
    axC.hlines(-85, x0, x0+15, colors='black') # add thick vertical bar from 100 to 115 ms at -85 mV , linewidth=4
    axC.text((x0 + x0+15) / 2, -100, '15 ms', color='black', ha='center', va='bottom')
    y0 = -85 
    scale_V_m_mV  = 15    # mV
    # voltage scalebar
    axC.plot([x0, x0], [y0, y0 + scale_V_m_mV], color='black', clip_on=False, zorder=5) # , lw=4.0
    axC.text(x0 - 2.0, y0 + scale_V_m_mV/2.0, f'{scale_V_m_mV} mV', ha='right', va='center', rotation=90, clip_on=False)

    
    axC.xaxis.set_visible(False)
    axC.yaxis.set_visible(False)
    axC.spines['left'].set_visible(False)
    axC.spines['bottom'].set_visible(False)
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])

    
    # D
    t_range = ionic_currents_adaptation["t_vec"]
    i_na_diff = ionic_currents_adaptation["I_Na"]
    i_ca_diff = ionic_currents_adaptation["I_Ca"]
    i_k_diff = ionic_currents_adaptation["I_K"]
    
    t_vec = t_range
    I_K, I_Na, I_Ca = i_k_diff, i_na_diff, i_ca_diff
    label_K, label_Na, label_Ca = '$I_{K^{+}}$', '$I_{Na^{+}}$', '$I_{Ca^{2+}}$'
    #color_1, color_2, color_3 = '#57e7ff', '#9357ff', '#ff1d1d'
    x_label, y_label = 'time (ms)', '$I_{ad} - I_{no\,ad}$ (nA)'
    description = None #'ionic currents during subthreshold fluctuations'
    pf.plot_ionic_current(t_vec, I_K, I_Na, I_Ca, label_K, label_Na, label_Ca, color_K, color_Na, color_Ca, x_label, y_label, description, ax=axD)
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])
    
    #plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S7_adaptation_currents.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    
    plt.show()

# Fig. S8
def fig_energy_budget_comparison(colors=['#57e7ff', '#9357ff', '#ff1d1d'], fontsizes={'panelletterfontsize': 15}, figsize=(7,6), savename_mode=False):
    # plots full comparison
    # input
    # colors are the colors for color_K, color_Na, color_h
    # fontsizes is a dictionary of used font sizes
    # savename_mode determines if the figures is saved or not
    
    color_K, color_Na, color_h = colors[0], colors[1], colors[2]
    
    V_Na = 50e-3 # in V
    V_K = -100e-3 # in V
    V_h = -43e-3 # in V 
    alpha = 0.05
    
    
    CTR=[92.7e6,-72.3e-3]
    FR=[113.4e6,-65.8e-3]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 5, height_ratios=[1, 2], width_ratios=[1, 1, 0.4, 1, 1], hspace=0.2, wspace=0.2)

    axA = fig.add_subplot(gs[0,0:2])
    axB1 = fig.add_subplot(gs[0,3])
    axB2 = fig.add_subplot(gs[0,4])
    axC = fig.add_subplot(gs[1,0:2])
    axD = fig.add_subplot(gs[1,3:5])

    ylim=None
    R_m = 9270000
    pf.plot_conductances(R_m, alpha, V_K, V_Na, V_h, colors=[color_K, color_Na, color_h], ax=axA)
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    pf.plot_energy_vs_V_RP(R_m, V_K, V_Na, V_h, alpha, ylim=ylim, ax=axB1)
    V_RP = -72.3e-3
    pf.plot_energy_vs_R_m(V_RP, V_K, V_Na, V_h, alpha, ylim=ylim, ax=axB2)
    axB2.set_ylabel("")
    panel_letter(axB1, "B", size=fontsizes['panelletterfontsize'])

    V_RP_vec, R_m_vec, E_RP_grid, E_RP_Attwell_grid = pf.compute_heatmap_data(V_K, V_Na, V_h, alpha)
    pf.plot_heatmap(E_RP_grid, CTR, FR, V_RP_vec, R_m_vec, '$E_{RP}$ with HCN', ax=axC)#, abs_scale=(0, 1.6e9))
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    pf.plot_heatmap(E_RP_Attwell_grid, CTR, FR, V_RP_vec, R_m_vec, '$E_{RP}$ without HCN', legend_mode=True, ax=axD)#, abs_scale=(0, 1.6e9))
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    if savename_mode is True:
        #plt.tight_layout()
        savepath='../Figures/0_paper/fig_S8_energy_budget_comparison.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        plt.show()
        
# Fig. S9
def fig_membrane_noise(baseline_signals, cleaned_signals, T, fs, results_single_runs_membrane_noise_CTR, results_single_runs_membrane_noise_FR, colors=['black', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(7,6), savename_mode=True):
    # create information figure
    # input
    # baseline_signals is a list of arrays of the raw and baseline cleaned signal (CTR & FR)
    # cleaned_signals is a list of arrays of the fully cleaned signal (CTR & FR)
    # T is the duration in ms
    # fs is the sampling frequency in Hz
    # results_single_runs_membrane_noise_CTR, results_single_runs_membrane_noise_FR are dictionaries of the CTR & FR simulated with different levels of membrane noise
    # colors are the colors for color_CTR, color_FR
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not 
    
    color_CTR, color_FR = colors[0], colors[1]
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 4, figure=fig, wspace=0.6, hspace=0.15, height_ratios=[1, 1, 0.4, 0.8, 0.2, 0.8] )

    baseline_signal_CTR, baseline_signal_FR = baseline_signals[0], baseline_signals[1]
    cleaned_signal_CTR, cleaned_signal_FR = cleaned_signals[0], cleaned_signals[1]

    # plot experimental data
    axAT = fig.add_subplot(gs[0, 0:2])
    pf.plot_raw_and_clean_V_m(baseline_signal_CTR, cleaned_signal_CTR, T, color=color_CTR, ax=axAT)
    panel_letter(axAT, "A", size=fontsizes['panelletterfontsize'])
    axAB = fig.add_subplot(gs[1, 0:2])
    pf.plot_raw_and_clean_V_m(baseline_signal_FR, cleaned_signal_FR, T, color=color_FR, ax=axAB)
    axBT = fig.add_subplot(gs[0, 2])
    pf.plot_power_spectrum(baseline_signal_CTR, cleaned_signal_CTR, fs, color=color_CTR, ax=axBT)
    axBT.set_xticklabels([])
    axBT.set_xlabel("")
    panel_letter(axBT, "B", size=fontsizes['panelletterfontsize'])
    axBB = fig.add_subplot(gs[1, 2])
    pf.plot_power_spectrum(baseline_signal_FR, cleaned_signal_FR, fs, color=color_FR, ax=axBB)
    axCT = fig.add_subplot(gs[0, 3])
    pf.plot_V_m_histograms(baseline_signal_CTR, cleaned_signal_CTR, color=color_CTR, ax=axCT)
    axCT.set_xticklabels([])
    axCT.set_xlabel("")
    panel_letter(axCT, "C", size=fontsizes['panelletterfontsize'])
    axCB = fig.add_subplot(gs[1, 3])
    pf.plot_V_m_histograms(baseline_signal_FR, cleaned_signal_FR, color=color_FR, ax=axCB)

    #axspace_1 = fig.add_subplot(gs[2, :])
    
    # plot simulation results
    axD = fig.add_subplot(gs[3, 0:2])
    pf.plot_membrane_noise_effect(results_single_runs_membrane_noise_CTR, results_single_runs_membrane_noise_FR, 'r_post', plot_mode='correlation_short', error_bars=False, mean_over_zeros=True, description_mode=False, colors=[color_CTR, color_FR], ax=axD)
    axD.set_xticklabels([])
    axD.set_xlabel("")
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    axE = fig.add_subplot(gs[3, 2:4])
    pf.plot_membrane_noise_effect(results_single_runs_membrane_noise_CTR, results_single_runs_membrane_noise_FR, 'E_tot', plot_mode='correlation_short', error_bars=False, mean_over_zeros=True, description_mode=False, colors=[color_CTR, color_FR], ax=axE)
    axE.get_legend().remove()
    axE.set_xticklabels([])
    axE.set_xlabel("")
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'])

    #axspace_2 = fig.add_subplot(gs[4, :])
    
    axF = fig.add_subplot(gs[5, 0:2])
    pf.plot_membrane_noise_effect(results_single_runs_membrane_noise_CTR, results_single_runs_membrane_noise_FR, 'OSI_per_energy', plot_mode='correlation_short', error_bars=False, mean_over_zeros=True, description_mode=False, colors=[color_CTR, color_FR], ax=axF)
    axF.get_legend().remove()
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])
    
    axG = fig.add_subplot(gs[5, 2:4])
    pf.plot_membrane_noise_effect(results_single_runs_membrane_noise_CTR, results_single_runs_membrane_noise_FR, 'MICE', plot_mode='correlation_short', error_bars=False, mean_over_zeros=True, description_mode=False, colors=[color_CTR, color_FR], ax=axG)
    axG.get_legend().remove()
    panel_letter(axG, "G", size=fontsizes['panelletterfontsize'])
    
    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S9_membrane_noise.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    
# Fig. S10
def fig_V_gap_additional(results_multiple_CTR_FR_runs_V_gap_AdExp, results_V_gap_variable_E_L, results_V_gap_variable_V_thresh, results_single_run_E_L_AdExp, results_single_run_V_thresh_AdExp, colors=['black', 'red', 'darkcyan', 'darkviolet'], fontsizes={'panelletterfontsize': 15}, figsize=(7,7), savename_mode=True):
    # create information figure
    # input
    # results_multiple_CTR_FR_runs_V_gap_AdExp is the dictionary of V_gap_constant results
    # results_V_gap_variable_E_L, results_V_gap_variable_V_thresh are the dictionary of V_gap_variable
    # results_single_run_E_L_AdExp, results_single_run_V_thresh_AdExp are the dictionary of V_gap_variable with different E_e & E_i
    # colors are the colors for color_CTR, color_FR, color_V_rest_var, color_V_thresh_var 
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not 
    
    color_CTR, color_FR, color_V_rest_var, color_V_thresh_var = colors[0], colors[1], colors[2], colors[3]
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(5, 4, figure=fig, wspace=0.6, hspace=0.4, height_ratios=[1, 1, 1, 0.05, 1.5])
    
    # plot experimental data
    axA = fig.add_subplot(gs[0, 0:2])
    pf.plot_V_gap_variable(results_V_gap_variable_E_L, results_V_gap_variable_V_thresh, 'r_post', mean_over_zeros=True, description_mode=False, plot_mode='correlation_short', colors=[color_V_rest_var, color_V_thresh_var], ax=axA)
    axA.set_xticklabels([])
    axA.set_xlabel("")
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=0.0)
    axB = fig.add_subplot(gs[0, 2:4])
    pf.plot_CTR_FR_V_gap_constant(results_multiple_CTR_FR_runs_V_gap_AdExp, ['r_post'], CTR_FR=False, shade_intersection=False, error_bars=False, mean_over_zeros=True, description_mode=False, plot_mode='correlation_short', colors=[color_CTR, color_FR], ax=axB)
    axB.set_xticklabels([])
    axB.set_xlabel("")
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=0.0)
    
    axC = fig.add_subplot(gs[1, 0:2])
    pf.plot_V_gap_variable(results_V_gap_variable_E_L, results_V_gap_variable_V_thresh, 'CV_V_m', mean_over_zeros=True, description_mode=False, plot_mode='correlation_short', colors=[color_V_rest_var, color_V_thresh_var], ax=axC)
    axC.get_legend().remove()
    axC.set_xticklabels([])
    axC.set_xlabel("")
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=0.0)
    axD = fig.add_subplot(gs[1, 2:4])
    pf.plot_CTR_FR_V_gap_constant(results_multiple_CTR_FR_runs_V_gap_AdExp, ['CV_V_m'], CTR_FR=False, shade_intersection=False, error_bars=False, mean_over_zeros=True, description_mode=False, plot_mode='correlation_short', colors=[color_CTR, color_FR], ax=axD)
    axD.get_legend().remove()
    axD.set_xticklabels([])
    axD.set_xlabel("")
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=0.0)

    axE = fig.add_subplot(gs[2, 0:2])
    pf.plot_V_gap_variable(results_V_gap_variable_E_L, results_V_gap_variable_V_thresh, 'MICE', mean_over_zeros=True, description_mode=False, plot_mode='correlation_short', colors=[color_V_rest_var, color_V_thresh_var], ax=axE)
    axE.get_legend().remove()
    axE.set_xlabel("$V_{thresh} - V_{rest}$ (mV)")
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=0.0)
    axF = fig.add_subplot(gs[2, 2:4])
    pf.plot_CTR_FR_V_gap_constant(results_multiple_CTR_FR_runs_V_gap_AdExp, ['MICE'], CTR_FR=False, shade_intersection=False, error_bars=False, mean_over_zeros=True, description_mode=False, plot_mode='correlation_short', colors=[color_CTR, color_FR], ax=axF)
    axF.get_legend().remove()
    axF.set_xlabel("$V_{rest}$ (mV)")
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=0.0)

    axG = fig.add_subplot(gs[4, 0:3])
    pf.plot_V_gap_variable_E_e_E_i(results_single_run_E_L_AdExp, results_single_run_V_thresh_AdExp, ['r_post'], plot_mode='correlation_short', colors=[color_V_rest_var, color_V_thresh_var], ax=axG)
    axG.legend(loc='center left', bbox_to_anchor=(0.9, 0.8), frameon=False)
    axG.set_title("")
    axG.set_xlabel("$V_{thresh} - V_{rest}$ (mV)")
    panel_letter(axG, "G", size=fontsizes['panelletterfontsize'], dx=-0.02, dy=0.0)
    
    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S10_V_gap_additional.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)


# Fig. S11
def fig_synaptic_input_Zeldenrust(t_stim, w_e_0, r_e, spike_times_e, spike_times_i, N_e_noise, T, tau_switch, colors=['#2e7d32', '#f9a825', '#f9a825', 'lightskyblue'], fontsizes={'panelletterfontsize': 15}, figsize=(5,8), savename_mode=True):
    # create figure on synaptic input structure and stimulation protocol for Zeldenrust input
    # input
    # t_stim is the array with the binary hidden state (0 or 1) over time
    # w_e_0 is an array of length N_e with normalized excitatory synaptic weights
    # r_e is a list of arrays with firing rates for excitatory neurons in Hz
    # spike_times_e, spike_times_i is a dictionary or list of spike times for excitatory & inhibitory neurons
    # N_e_noise is the number excitatory noise synapses
    # T is the simulation duration in ms
    # tau_switch is the time constant in ms defining hidden state switching probability
    # colors are the colors for color_w_e, color_r_e, color_input_exc, color_input_inh
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    color_w_e, color_r_e, color_input_exc, color_input_inh = colors[0], colors[1], colors[2], colors[3]
    
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = GridSpec(4,2, figure=fig, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0, 1.0, 2.0])#, wspace=1.0, hspace=0.3)  

    w_e = pf.w_scale_to_w_e_syn(w_e_0, len(w_e_0))
    w_e = w_e / np.mean(w_e)
    
    # A: stimulation protocol image
    time = np.arange(T) / 1000.0  # ms to seconds
    
    
    # A: excitatory synaptic weights & firing rates distribution
    axA_w = fig.add_subplot(gs[0, 0])
    axA_r = fig.add_subplot(gs[1, 0])

    # get signalling parts
    w_e_signal = w_e[N_e_noise:]
    r_e_signal = r_e[N_e_noise:]

    pf.plot_one_histogram(w_e, 'exc weights (nS)', '$n_{syn}$', '$w_{e}$', description=None, color=color_w_e, axis_mode='log', ax=axA_w)
    axA_w.get_legend().remove() if axA_w.get_legend() is not None else None
    pf.plot_one_histogram(r_e[:,0], 'exc rates  (Hz)', '$n_{syn}$', '$r_{e}$', description=None, color=color_r_e, axis_mode='linear', ax=axA_r)
    axA_r.get_legend().remove() if axA_r.get_legend() is not None else None
    panel_letter(axA_w, "A", size=fontsizes['panelletterfontsize'])
    
    # B: excitatory synaptic weights & firing rates matching logic for different stimul
    axB_OFF  = fig.add_subplot(gs[0, 1])
    axB_ON = fig.add_subplot(gs[1, 1])

    idx_t_OFF = np.where(np.isclose(t_stim, 0.0))[0][0]
    idx_t_ON = np.where(np.isclose(t_stim, 1.0))[0][0]
    
    r_e_signal_OFF = r_e_signal[:, idx_t_OFF]  
    r_e_signal_ON = r_e_signal[:, idx_t_ON] 
    pf.plot_synapse_weights_and_rates(w_e_signal, r_e_signal_OFF, description='Synaptic input OFF-stimulus', colors=[color_w_e, color_r_e], ax=axB_ON)
    pf.plot_synapse_weights_and_rates(w_e_signal, r_e_signal_ON, description='Synaptic input ON-stimulus', colors=[color_w_e, color_r_e], ax=axB_OFF)
    panel_letter(axB_OFF, "B", size=fontsizes['panelletterfontsize'])

    axC = fig.add_subplot(gs[2, :])
    axC.step(time, t_stim, where='post', color='black', lw=1.5)
    axC.set_ylabel("hidden state (0/1)")
    axC.set_title(f"Hidden state and excitatory input (switching input, $\\tau_{{switch}} = {round(tau_switch,0)}\\ \\mathrm{{ms}}$)")
    axC.set_yticks([0, 1])
    axC.set_ylim(-0.1, 1.1)
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    axC.tick_params(axis='x', labelbottom=False,  bottom=True)
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    #axC.text(-0.02, 1.05, 'C', transform=axA.transAxes, fontweight='bold', va='bottom', ha='left')

    # D: raster plot
    axD = fig.add_subplot(gs[3, :])

    pf.plot_raster(spike_times_e, spike_times_i, N_e_noise, orientation_mode=False, title_mode=False, colors=[color_input_exc, color_input_inh], ax=axD) #  w_e=w_e for w_e dependent alpha
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    
    #plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S11_synaptic_input_Zeldenrust.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True) 
        
    plt.show()
    
    
# Fig. S12 
def fig_cornerplot_with_voltage_trace(means, stds, maps, samples, t_vecs, V_ms, spike_times_posts, r_posts, param_names, param_units, t_ms=None, V_m_mean=None, V_m_samples=None, description=None, savename_mode=True):
    # create corner plot of variable size with experimental vs simulated voltage trace & spike times in the top row
    
    # input
    # means, stds, maps, samples are the means, standard deviation, maps and samples of the SBI-inferenced parameters
    # t_vecs is a list of the experimental & SBI-based simulated time vectors in ms 
    # V_ms is a list of the experimental & SBI-based simulated membrane voltages in mV
    # spike_times_posts is a list of the experimental & SBI-based simulated spike times in ms
    # r_posts is a list of the experimental & SBI-based simulated firing rates in Hz
    # param_names is a list of the parameters names 
    # param_units is a list of the parameter units 
    # t_ms is the time vector in ms
    # V_m_mean are mean voltage values in mV
    # V_m_samples are n_samples standard deviation traces in mV
    # description is the title of the figure
    # savename_mode decides whether the figure is saved or not
    
    color_SBI_mean = "slateblue"
    color_SBI_map = "mediumorchid"
    color_exp = "grey"
    n_params = samples.shape[1]
    assert n_params == len(param_names) == len(param_units) == len(maps) == len(means) == len(stds), "Mismatch in parameter dimensions"

    fig = plt.figure(figsize=(3 * n_params, 3 * n_params + 2))
    outer_gs = gs.GridSpec(n_params + 1, n_params, figure=fig, height_ratios=[1.2] + [1]*n_params)

    # voltage trace on top spanning all columns
    ax_trace = fig.add_subplot(outer_gs[0, :])
    ax_trace.plot(t_vecs[1], V_ms[0], label=r"exp $V_m$", color = color_exp)
    ax_trace.plot(t_vecs[1], V_ms[1], label=r"SBI $V_m$", color = color_SBI_mean)
    ax_trace.scatter(spike_times_posts[0], np.ones_like(spike_times_posts[0]) * 1.01 * (max([max(V_ms[0]),max(V_ms[1])])+4), label=f"exp spikes, {r_posts[0]} Hz", s=10, color = color_exp)
    ax_trace.scatter(spike_times_posts[1], np.ones_like(spike_times_posts[1]) * 0.99 * (max([max(V_ms[0]),max(V_ms[1])])+4), label=f"SBI spikes, {r_posts[1]} Hz", s=10, color = color_SBI_mean)
    ax_trace.set_title("Experimental and simulation-based inferred voltage trace comparison") # "Smoothed voltage trace comparison"
    ax_trace.set_xlabel("Time / ms")
    ax_trace.set_ylabel(r"$V_m$ / mV")
    ax_trace.spines['top'].set_visible(False)
    ax_trace.spines['right'].set_visible(False)
    ax_trace.legend(loc='upper right', facecolor='white', edgecolor='none', framealpha=1.0, fontsize='small')

    # cornerplot (marginals and 2D joints)
    for i in range(n_params):
        for j in range(n_params):
            ax = fig.add_subplot(outer_gs[i+1, j])
            if i == j:
                # marginal
                sns.kdeplot(samples[:, j], ax=ax, fill=True)
                ax.axvline(maps[j], color=color_SBI_map, linestyle='--')
                ax.axvline(means[j], color=color_SBI_mean, linestyle='-.')
                ax.set_yticks([])
                ax.set_xlabel(param_names[j])
                ax.set_title(f"{param_names[j]}\nMAP {maps[j]:.2f} {param_units[j]}\nMean {means[j]:.2f} ± {stds[j]:.2f} {param_units[j]}")
            elif i > j:
                # lower triangle: 2D joint plot
                sns.kdeplot(x=samples[:, j], y=samples[:, i], ax=ax, fill=True, cmap="viridis", levels=100)
                ax.plot(maps[j], maps[i], 'o', color=color_SBI_map)
                ax.plot(means[j], means[i], 'o', color= color_SBI_mean)
                ax.set_xlabel(param_names[j])
                ax.set_ylabel(param_names[i])
            else:
                # upper triangle: leave blank
                ax.axis("off")

    # plot V_m_posterior_sample mean & std plot
    if V_m_samples is not None:
        ax_samples = fig.add_subplot(outer_gs[1:4, -3:]) 
        pf.plot_V_m_SBI_posterior_samples(t_ms, V_m_mean, V_m_samples, colors=[color_SBI_mean, "#4C72B0"], scalebar_width=2.0, ylims=None,  ax=ax_samples)
        ax_samples.set_title("Simulated voltage trace of mean inferred parameters with 1 standard deviation") # "Smoothed voltage trace comparison"
        
    # Legend
    legend_ax = fig.add_subplot(outer_gs[-1, -1])
    legend_ax.axis("off")
    legend_elements = [
        Line2D([0], [0], color=color_SBI_map, linestyle='--', label='MAP'),
        Line2D([0], [0], color=color_SBI_mean, linestyle='-.', label='Mean')]
    legend_ax.legend(handles=legend_elements, facecolor='white', edgecolor='none', framealpha=1.0, loc='upper left', frameon=True)

    fig.suptitle(f"Posterior distributions, Experimental and simulation-based-inferred Voltage Traces of {description}", y=1.02)
    plt.tight_layout()
    
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S12_SBI_cornerplot_Zeldenrust.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True) 
    # OLD naming
    #if savename:
        #plt.savefig(f"../Figures/{savename}.pdf", bbox_inches='tight')
    plt.show()
    
    
def fig_cornerplot_correlations_Zeldenrust(results_exc, results_inh, names_list, description=None, colors=['red', 'blue'], savename_mode=True):
    # create corner plot of variable size with correlations
    
    # input
    # results_exc, results_inh are the dictionaries containing the data
    # names_list is the list of keys to plot
    # description is an optional description
    # colors are the colors for exc & inh cells
    # description is the title of the figure
    # savename_mode decides whether the figure is saved or not

    n_params = len(names_list)
    
    fig = plt.figure(figsize=(3 * n_params, 3 * n_params))
    grid = gs.GridSpec(n_params , n_params, figure=fig)

    # cornerplot of correlations
    for i in range(n_params):
        for j in range(n_params):
            ax = fig.add_subplot(grid[i, j])
            if i > j:
                # lower triangle correlation plots
                x_name = names_list[i]
                y_name = names_list[j]
                x_label = pf.value_key_text_plot_Zeldenrust(names_list[i], plot_mode='correlation')
                y_label = pf.value_key_text_plot_Zeldenrust(names_list[j], plot_mode='correlation')
                pf.plot_correlation_exc_inh(results_exc[x_name], results_inh[x_name], results_exc[y_name], results_inh[y_name], x_label, y_label, colors=colors, ax=ax)
                ax.legend_.remove()
            else:
                # leave upper triangle & diagonal blank
                ax.axis("off")

    if description is not None: 
        fig.suptitle(f"{description}")
    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_S13_data_correlation_Zeldenrust.pdf"
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True) 
    # OLD naming
    #if savename:
        #plt.savefig(f"../Figures/{savename}_correlation_cornerplot.pdf", bbox_inches='tight')
    plt.show()
    
def fig_3x3_pc_loadings_bars(pca_all, pca_exc, pca_inh, feature_names, colors=['grey', 'coral', 'darkcyan'], description=False, figsize=(7,10), savename_mode=True, fontsizes={'panelletterfontsize': 15}):
    # plot horizontal bar plots of PCA loadings for first 3 PCs in 3 rows (all/exc/inh)
    # input
    # pca_all, pca_exc, pca_inh are fitted PCA objects
    # feature_names is a list of the feature names
    # colors are the color to plot the bars in
    # description is a string for the overall title of the plot
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    # fontsizes is a dictionary of used font sizes
    
    # output
    # figure with 3x3 panels (rows: all/exc/inh, cols: PC1/PC2/PC3)

    pcs = ('PC1', 'PC2', 'PC3')
    pcas = [pca_all, pca_exc, pca_inh]
    row_titles = ['all', 'exc', 'inh']
    row_panel_letters = ['A', 'B', 'C']

    fig = plt.figure(figsize=figsize)
    grid = gs.GridSpec(len(pcas), len(pcs), figure=fig, wspace=0.7, hspace=0.5)
    axes = [[fig.add_subplot(grid[r, c]) for c in range(len(pcs))] for r in range(len(pcas))]

    for r, (pca, row_name, row_letter, color) in enumerate(zip(pcas, row_titles, row_panel_letters, colors)):
        for c, pc in enumerate(pcs):
            ax = axes[r][c]

            # plot loadings into the given axis
            pf.plot_pc_loadings_bars(pca, feature_names, pc=pc, color=color, ax=ax)

            # add row label on the leftmost panel only
            if c == 0:
                # panel letter for each row (A, B, C)
                panel_letter(ax, row_letter, size=fontsizes.get('panelletterfontsize', 15))

        # row title centered above row 
        mid_ax = axes[r][1]  # PC2 axis
        bbox = mid_ax.get_position()
        x_center = 0.5 * (bbox.x0 + bbox.x1)
        y_top = bbox.y1 + 0.025

        fig.text(x_center, y_top, row_name, ha='center', va='bottom', color=color, fontsize=fontsizes.get('rowtitlefontsize', 13), fontweight='bold')
    
    # overall figure title
    if description:
        fig.suptitle(description, y=0.995)

    plt.tight_layout()

    if savename_mode is True:
        savepath = "../Figures/0_paper/fig_S14_PC_loadings_full.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)

    plt.show()
    return fig, axes

    

############################## bin ##############################

def OLD_fig_Pareto_optimality(figA, figB, figC, results_mean_spiking_trials, results_grid_point_to_exp_data=None, fontsizes={'panelletterfontsize': 15}, figsize=(9,7), savename_mode=True): #, figsize=(13, 11)
    # create Pareto_optimality figure
    # input
    # figA, figB, figC are the 3D figures
    # results_mean_spiking_trials is a dictionary of the full grid results
    # results_grid_point_to_exp_data is a dictionary of the experimental data points fitted to their closest grid point 
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    fig, axes = plt.subplots(3, 3, figsize=figsize, dpi=300)
    
    axes = axes.ravel()

    # A–C 
    
    render_3Dfig_to_ax(figA, axes[0])
    panel_letter(axes[0], "A", size=fontsizes['panelletterfontsize'])
    
    render_3Dfig_to_ax(figB, axes[1])
    panel_letter(axes[1], "B", size=fontsizes['panelletterfontsize'])
    
    render_3Dfig_to_ax(figC, axes[2])
    panel_letter(axes[2], "C", size=fontsizes['panelletterfontsize'])

    # D -- I
    R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy = pf.load_data_correlations(results_mean_spiking_trials)

    # D
    pf.plot_correlation(E_tot, MI, 'E_tot', 'MI', MI_per_energy, 'MI_per_energy', results_grid_point_to_exp_data=None, inverted_x=True, plot_mode='correlation_short', ax=axes[3])
    axes[3].xaxis.get_offset_text().set_y(2.0)
    panel_letter(axes[3], "D", size=fontsizes['panelletterfontsize'])

    # E
    pf.plot_correlation(E_tot, OSI, 'E_tot', 'OSI', OSI_per_energy, 'OSI_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, inverted_x=True, plot_mode='correlation_short', ax=axes[4])
    axes[4].xaxis.get_offset_text().set_y(20.0)
    panel_letter(axes[4], "E", size=fontsizes['panelletterfontsize'])

    # F
    pf.plot_correlation(E_tot, TE, 'E_tot', 'TE', TE_per_energy, 'TE_per_energy', results_grid_point_to_exp_data=None, inverted_x=True, plot_mode='correlation_short', ax=axes[5])
    axes[5].xaxis.get_offset_text().set_y(0.0001)
    panel_letter(axes[5], "F", size=fontsizes['panelletterfontsize'])

    # G: fit highlighted points of MI_per_energy
    fit_func_highlight=pf.linear_rate_information_func # options: linear_func 2, sqrt_func 2, log_func 3, rate_information_func 3, linear_rate_information_func 5, sqrt_rate_information_func 5, linear_log_func 4, piecewise_linear_exponential_func 3
    initial_guess_highlight=[1,1,1,1,1]
    
    R_m_init, E_L_init = 150.0, -50.0
    x_label, y_label = 'r_post', 'MI_per_energy'
    w_scale_list_x, x_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, x_label, R_m_init, E_L_init)
    w_scale_list_y, y_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, y_label, R_m_init, E_L_init)
    highlight_points = (x_highlights, y_highlights)
    # z: E_tot, 'E_tot'
    pf.plot_correlation(r_post, MI_per_energy, x_label, y_label, highlight_points=highlight_points, fit_func_highlight=fit_func_highlight, initial_guess_highlight=initial_guess_highlight, plot_mode='correlation_short', ax=axes[6])
    axes[6].get_legend().remove()
    panel_letter(axes[6], "G", size=fontsizes['panelletterfontsize'])

    # H
    pf.plot_correlation(r_post, MICE, 'r_post', 'MICE', MICE_per_energy, 'MICE_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, plot_mode='correlation_short', ax=axes[7])
    axes[7].set_xlim(right=25) 
    axes[7].get_legend().remove()
    panel_letter(axes[7], "H", size=fontsizes['panelletterfontsize'])

    # I (same correlation style but different layout/axis if desired)
    pf.plot_correlation(r_post, MICE, 'r_post', 'MICE', MICE_per_energy, 'MICE_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, plot_mode='correlation_short', ax=axes[8])
    axes[8].get_legend().remove()
    panel_letter(axes[8], "I", size=fontsizes['panelletterfontsize'])

    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_1_Pareto_optimality.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True, dpi=1800)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    #return fig

# large version of Fig. 2
def OLD_fig_multiple_trajectories_examples(fig3d, data_CTR_like, data_FR_like, color_CTR_like='#4a61e8', color_FR_like='#c41c5f', fontsizes={'panelletterfontsize': 15}, figsize=(10, 10), savename_mode=True): #, figsize=(15, 13)
    # Compose the 3D + two examples figure
    # input
    # fig3d is the 3D figure (with trajectories)
    # data_CTR_like, data_FR_like are dictionaries of the CTR_like & FR_like points
    # color_CTR_like, color_FR_like are the colors of the CTR_like & FR_like points
    # fontsizes is a dictionary of used font sizes
    # figsize is the figure size
    # savename_mode decides whether fig is saved or not

    # create 3D panel
    
    # add CTR_like & FR_like markers at their locations
    R_m_CTR_like, E_L_CTR_like, w_scale_CTR_like = data_CTR_like['R_m'], data_CTR_like['E_L'], data_CTR_like['w_scale']
    R_m_FR_like, E_L_FR_like, w_scale_FR_like = data_FR_like['R_m'], data_FR_like['E_L'], data_FR_like['w_scale']
    
    w_e_CTR_like = float(pf.w_scale_to_w_e_syn(w_scale_CTR_like)) # nS
    w_e_FR_like = float(pf.w_scale_to_w_e_syn(w_scale_FR_like)) # nS

    fig3d.add_trace(go.Scatter3d(x=[R_m_CTR_like], y=[E_L_CTR_like], z=[w_e_CTR_like], mode='markers', marker=dict(size=15, color=color_CTR_like, line=dict(width=1, color='white')), showlegend=False))
    fig3d.add_trace(go.Scatter3d(x=[R_m_FR_like], y=[E_L_FR_like], z=[w_e_FR_like], mode='markers', marker=dict(size=15, color=color_FR_like, line=dict(width=1, color='white')), showlegend=False))
    fig3d.update_layout(showlegend=False)

    # save 3D fig as png to reload it later
    three_d_png_path = '../Figures/0_paper/fig_single_trajectory_examples_3d.png'  
    pio.write_image(fig3d, three_d_png_path, format='png', scale=2)
        
    # plot simulated CTR_like & FR_like data
    V_m_CTR_like = data_CTR_like['V_m']
    V_m_FR_like = data_FR_like['V_m']
    
    V_m_shifted_CTR_like, Delta_V_m_CTR_like = pf.shift_V_m(V_m_CTR_like, target_mV=0.0)
    V_m_shifted_FR_like, Delta_V_m_FR_like = pf.shift_V_m(V_m_FR_like, target_mV=0.0)
    
    V_m_excerpts_CTR_like, V_m_min_CTR_like, V_m_max_CTR_like, V_m_mean_CTR_like = pf.cut_into_excerpts(V_m_shifted_CTR_like, n_excerpts=9, start_idx=3000, stop_idx=30000)
    V_m_excerpts_FR_like, V_m_min_FR_like, V_m_max_FR_like, V_m_mean_FR_like = pf.cut_into_excerpts(V_m_shifted_FR_like, n_excerpts=9, start_idx=3000, stop_idx=30000)

    # parameters of CTR_like & FR_like points 
    CTR_like_txt = (rf'$R_m={R_m_CTR_like:.0f}\,\mathrm{{M\Omega}},\ V_{{rest}}={E_L_CTR_like:.0f}\,\mathrm{{mV}},$' + '\n' + rf'$\langle w_{{syn,e}}\rangle={w_e_CTR_like:.2f}\,\mathrm{{nS}}$')
    FR_like_txt = (rf'$R_m={R_m_FR_like:.0f}\,\mathrm{{M\Omega}},\ V_{{rest}}={E_L_FR_like:.0f}\,\mathrm{{mV}},$' + '\n' + rf'$\langle w_{{syn,e}}\rangle={w_e_FR_like:.2f}\,\mathrm{{nS}}$')
    
    # compose final fig
    fig = plt.figure(figsize=figsize)
    G = gs.GridSpec(2, 3, figure=fig, height_ratios=[4.5, 1.5], wspace=0.15, hspace=0.00)
    fig.subplots_adjust(top=0.98, bottom=0.07, left=0.06, right=0.99)

    # top: 3D fig
    ax_top = fig.add_subplot(G[0, :])
    im = Image.open(three_d_png_path).convert("RGBA")
    bbox = im.getbbox()  # get nontransparent bounding box
    im_cropped = im.crop(bbox)
    im_array = np.array(im_cropped)
    ax_top.imshow(im_array)
    ax_top.axis('off')
    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # cut manually
    #ax_top.imshow(mpimg.imread(three_d_png))
    #ax_top.set_xlim(100, 1200)   # cut horizontal padding
    #ax_top.set_ylim(700, 100)    # cut vertical padding (reverse since image origin at top)

    # bottom: CTR_like & FR_like points
    V_m_min, V_m_max = pf.compute_shared_ylim(V_m_min_CTR_like, V_m_max_CTR_like, V_m_min_FR_like, V_m_max_FR_like)
    
    # bottom left: FR_like point
    ax_left = fig.add_subplot(G[1, 0])
    pf.plot_V_m_excerpts(V_m_excerpts_FR_like, color=color_FR_like, ylims=(V_m_min, V_m_max), ax=ax_left)
    ax_left_bbox = ax_left.get_position()
    fig.text((ax_left_bbox.x0 + ax_left_bbox.x1) / 2.0, ax_left_bbox.y0 - 0.025, FR_like_txt, ha='center', va='top') #, fontsize=20

    # bottom middle: histogram
    xmin = -45 
    xmax = -80 
    V_thresh = data_CTR_like['V_thresh']
    ax_mid = fig.add_subplot(G[1, 1])
    ax_mid.hist(V_m_FR_like, label='FR like', alpha=0.8, bins=np.linspace(xmax, xmin, 80), color=color_FR_like, density=True)
    ax_mid.hist(V_m_CTR_like, label='CTR like', alpha=0.8, bins=np.linspace(xmax, xmin, 80), color=color_CTR_like, density=True)
    #pf.plot_two_histograms(V_m_FR_like, V_m_CTR_like, x_label=r'$V_m$ (mV)', y_label=None, label_1='FR like', label_2='CTR like', description=None, color_1=color_FR_like, color_2=color_CTR_like, bins=np.linspace(xmax, xmin, 80), density=True, ax=ax_mid)
    ax_mid.set_xlim(xmin, xmax)
    ax_mid.tick_params(axis='x') # , labelsize=20
    ax_mid.set_ylim(0, 0.3)
    #ax_mid.axis('off')
    ax_mid.axvline(V_thresh, 0, 0.85, color=color_CTR, ls='--', alpha=0.9)
    ax_mid.text(V_thresh, 0.27, r'$V_{\mathrm{thresh}}$', ha='center', va='bottom') 
    ax_mid.set_xlabel(r'$V_m$ (mV)') 
    #ax_mid.set_ylabel('density')
    #ax_mid.set_ylabel(None)
    ax_mid.yaxis.set_visible(False)
    ax_mid.spines['left'].set_visible(False)
    ax_mid.xaxis.set_inverted(True) 
    ax_mid.spines['top'].set_visible(False)
    ax_mid.spines['right'].set_visible(False)

    # bottom right: CTR_like point
    ax_right  = fig.add_subplot(G[1, 2])
    pf.plot_V_m_excerpts(V_m_excerpts_CTR_like, color=color_CTR_like, ylims=(V_m_min, V_m_max), ax=ax_right)
    ax_right_bbox = ax_right.get_position()
    fig.text((ax_right_bbox.x0 + ax_right_bbox.x1) / 2.0, ax_right_bbox.y0 - 0.025, CTR_like_txt, ha='center', va='top')
    
    # panel letters
    top_bbox  = ax_top.get_position()
    left_bbox = ax_left.get_position()
    mid_bbox  = ax_mid.get_position()
    right_bbox= ax_right.get_position()
    
    
    fig.text(top_bbox.x0  - 0.03, top_bbox.y1  - 0.15, 'A', fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='top') 
    fig.text(left_bbox.x0 - 0.03, left_bbox.y1 - 0.02, 'B', fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='top') 
    fig.text(mid_bbox.x0  - 0.03, mid_bbox.y1  - 0.02, 'C', fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='top') 
    fig.text(right_bbox.x0- 0.03, right_bbox.y1 - 0.02, 'D', fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='top') 
    #panel_letter(ax_top, "A")
    #panel_letter(ax_left, "B")
    #panel_letter(ax_mid, "C")
    #panel_letter(ax_right, "D")
    
    # save & show fig
    savepath='../Figures/0_paper/fig_single_trajectory_examples.pdf'
    fig.savefig(savepath, bbox_inches='tight', transparent=True)
    plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)

# large version of Fig. 3
def fig_etiology_tuning_curve_broadening_large(tuning_curves_CTR_LIF, tuning_curves_FR_LIF, tuning_curves_CTR_LIFad, tuning_curves_FR_LIFad, tuning_curves_CTR_LIFexp, tuning_curves_FR_LIFexp, tuning_curves_CTR_AdExp, tuning_curves_FR_AdExp, mEPSC_avg_norm_CTR, mEPSC_avg_norm_FR, lognormal_function, tuning_curves_ms_CTR, tuning_curves_ms_FR, tuning_curves_nms_CTR, tuning_curves_nms_FR, results_membrane_noise_CTR, results_membrane_noise_FR, fontsizes={'panelletterfontsize': 15}, figsize=(7,5), savename_mode=False):
    # compose figure about the etiology of tuning curve
    
    # input
    # (A–D) LIF/LIF+Ad/LIF+Exp/AdExp:
    # tuning_curves_CTR, tuning_curves_FR are lists or arrays of the CTR & FR tuning curves
    # (E–G) multiplicative vs non-multiplicative scaling
    # mEPSC_avg_norm_CTR/FR are arrays of experimental mEPSC distributions
    # lognormal_function is the fitting function for the mEPSC distributions
    # tuning_curves_ms_CTR/FR are multiplicative-scaling tuning curves
    # tuning_curves_nms_CTR/FR are non-multiplicative scaling tuning curves
    # effect of membrane noise (H–J):
    # results_membrane_noise_CTR/FR are dictionaries containing experimental membrane noise level data
    # layout notes:
    # fontsizes is a dictionary of used font sizes
    # savename_mode determines if the figures is saved or not

    fig = plt.figure(figsize=figsize)
    # 3 rows x 5 columns; keep last column mostly unused to allow a clean 2-col span for OSI
    gs = GridSpec(3, 5, figure=fig, width_ratios=[1,1,1,1,1], height_ratios=[1.0, 1.0, 1.0], wspace=0.35, hspace=0.7)

    # (A–D) LIF/LIF+Ad/LIF+Exp/AdExp:
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # A: LIF
    axA = axes_top[0]
    pf.plot_tuning_curves(tuning_curves_CTR_LIF, tuning_curves_FR_LIF, label_CTR='CTR', label_FR='FR', normalized=True, mean_over_zeros=False, mode='tuning_curve', show_legend=False, show_xlabel=True, show_ylabel=True, ax=axA)
    axA.set_title("LIF")
    axA.set_xlabel('Distance from \n Preferred ($\circ$)')
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])

    # B: LIF+Ad
    axB = axes_top[1]
    pf.plot_tuning_curves(tuning_curves_CTR_LIFad, tuning_curves_FR_LIFad, label_CTR='CTR', label_FR='FR', normalized=True, mean_over_zeros=False, mode='tuning_curve', show_legend=False, show_xlabel=True, show_ylabel=False, ax=axB)
    axB.set_title("LIF+Ad")
    axB.set_xlabel('Distance from \n Preferred ($\circ$)')
    axB.set_yticklabels([])
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    # C: LIF+Exp
    axC = axes_top[2]
    pf.plot_tuning_curves(tuning_curves_CTR_LIFexp, tuning_curves_FR_LIFexp, label_CTR='CTR', label_FR='FR', normalized=True, mean_over_zeros=False, mode='tuning_curve', show_legend=False, show_xlabel=True, show_ylabel=False, ax=axC)
    axC.set_title("LIF+Exp")
    axC.set_xlabel('Distance from \n Preferred ($\circ$)')
    axC.set_yticklabels([])
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])

    # D: AdExp
    axD = axes_top[3]
    pf.plot_tuning_curves(tuning_curves_CTR_AdExp, tuning_curves_FR_AdExp, label_CTR='CTR', label_FR='FR', normalized=True, mean_over_zeros=False, mode='tuning_curve', show_legend=False, show_xlabel=True, show_ylabel=False, ax=axD)
    axD.set_title("AdExp")
    axD.set_xlabel('Distance from \n Preferred ($\circ$)')
    axD.set_yticklabels([])
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    # (E-G) synaptic scaling data: 
    # E: mEPSC lognormal fit
    axE = fig.add_subplot(gs[1, 0:2])
    pf.plot_mEPSC_lognormal_fit(mEPSC_avg_norm_CTR, mEPSC_avg_norm_FR, lognormal_function=lognormal_function, label_mode="short", scale_factor=1.845, bins=500, color_CTR=color_CTR, color_FR=color_FR, color_scaled='blue', title=None, ax=axE)
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'])

    # F: Simulation (mult. scaling)
    axF = fig.add_subplot(gs[1, 2])
    pf.plot_tuning_curves(tuning_curves_ms_CTR, tuning_curves_ms_FR, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR='blue', show_legend=False, show_xlabel=True, show_ylabel=True, ax=axF)
    axF.set_title('AdExp \n (mult.)') # scaled
    axF.set_xlabel('Distance from \n Preferred ($\circ$)')
    axF.set_yticklabels([])
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])

    # G: Simulation (non-mult. scaling)
    axG = fig.add_subplot(gs[1, 3])
    pf.plot_tuning_curves(tuning_curves_nms_CTR, tuning_curves_nms_FR, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR, show_legend=False, show_xlabel=True, show_ylabel=False, ax=axG)
    axG.set_title('AdExp \n (non-mult.)') # scaled
    axG.set_xlabel('Distance from \n Preferred ($\circ$)')
    axG.set_yticklabels([])
    panel_letter(axG, "G", size=fontsizes['panelletterfontsize'])

    # (H-J) membrane noise effect
    # H: no membrane noise (sigma = 0) tuning curve
    axH = fig.add_subplot(gs[2, 0])
    noise_index_0 = 0
    tcs_CTR_0 = [np.array(tc[noise_index_0]) for tc in results_membrane_noise_CTR['tuning_curve']]
    tcs_FR_0  = [np.array(tc[noise_index_0]) for tc in results_membrane_noise_FR['tuning_curve']]
    pf.plot_tuning_curves(tcs_CTR_0, tcs_FR_0, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR, show_legend=False, show_xlabel=True, show_ylabel=True, ax=axH)
    axH.set_title('AdExp \n' + r'$\sigma=0$ mV/ms')
    axH.set_xlabel('Distance from \n Preferred ($\circ$)')
    panel_letter(axH, "H", size=fontsizes['panelletterfontsize'])

    # I: OSI for different membrane noise levels
    axI = fig.add_subplot(gs[2, 1:3])
    pf.plot_membrane_noise_effect(results_membrane_noise_CTR, results_membrane_noise_FR, 'OSI', error_bars=True, colors=[color_CTR, color_FR], ax=axI)
    axI.set_title('OSI for different membrane noise levels \n ')
    panel_letter(axI, "I", size=fontsizes['panelletterfontsize'])

    # J: strong membrane noise (sigma = 8) tuning curve
    axJ = fig.add_subplot(gs[2, 3])
    noise_index_8 = 10
    tcs_CTR_8 = [np.array(tc[noise_index_8]) for tc in results_membrane_noise_CTR['tuning_curve']]
    tcs_FR_8  = [np.array(tc[noise_index_8]) for tc in results_membrane_noise_FR['tuning_curve']]
    pf.plot_tuning_curves(tcs_CTR_8, tcs_FR_8, 'CTR', 'FR', normalized=True, color_CTR=color_CTR, color_FR=color_FR, show_legend=False, show_xlabel=True, show_ylabel=True, ax=axJ)
    axJ.set_title('AdExp \n' + r'$\sigma=8$ mV/ms')
    axJ.set_xlabel('Distance from \n Preferred ($\circ$)')
    axJ.set_ylabel(None)
    panel_letter(axJ, "J", size=fontsizes['panelletterfontsize'])

    if savename_mode is True:
        #plt.tight_layout()
        savepath='../Figures/0_paper/fig_etiology_tuning_curve_broadening.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        print(f"saved: {savepath}")
        plt.show()

    plt.show()
    
def OLD_fig_energy_budget_comparison(fontsizes={'panelletterfontsize': 15}, figsize=(12, 8), savename_mode=False):
    # plots full comparison
    # input
    # fontsizes is a dictionary of used font sizes
    # savename_mode determines if the figures is saved or not
    
    V_Na = 50e-3 # in V
    V_K = -100e-3 # in V
    V_h = -43e-3 # in V 
    alpha = 0.05
    
    
    CTR=[92.7e6,-72.3e-3]
    FR=[113.4e6,-65.8e-3]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 8, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1, 1, 1], hspace=0.5, wspace=1.2)

    ax1 = fig.add_subplot(gs[0, :3])
    ax2 = fig.add_subplot(gs[0, 4:6])
    ax3 = fig.add_subplot(gs[0, 6:])
    ax4 = fig.add_subplot(gs[1:3, 0:4])
    ax5 = fig.add_subplot(gs[1:3, 4:])

    R_m = 9270000
    pf.plot_conductances(R_m, alpha, V_K, V_Na, V_h, ax=ax1)
    panel_letter(ax1, "A", size=fontsizes['panelletterfontsize'])
    pf.plot_energy_vs_V_RP(R_m, V_K, V_Na, V_h, alpha, ax=ax2)
    V_RP = -72.3e-3
    pf.plot_energy_vs_R_m(V_RP, V_K, V_Na, V_h, alpha, ax=ax3)
    panel_letter(ax2, "B", size=fontsizes['panelletterfontsize'])

    V_RP_vec, R_m_vec, E_RP_grid, E_RP_Attwell_grid = pf.compute_heatmap_data(V_K, V_Na, V_h, alpha)
    pf.plot_heatmap(E_RP_grid, CTR, FR, V_RP_vec, R_m_vec, 'ATP HCN', ax=ax4)#, abs_scale=(0, 1.6e9))
    panel_letter(ax4, "C", size=fontsizes['panelletterfontsize'])
    pf.plot_heatmap(E_RP_Attwell_grid, CTR, FR, V_RP_vec, R_m_vec, 'ATP No HCN', legend_mode=True, ax=ax5)#, abs_scale=(0, 1.6e9))
    panel_letter(ax5, "D", size=fontsizes['panelletterfontsize'])

    if savename_mode is True:
        #plt.tight_layout()
        save_path='../Figures/0_paper/fig_S8_energy_budget_comparison.pdf'
        fig.savefig(save_path, bbox_inches='tight', transparent=True)
        print(f"saved: {save_path}")
        plt.show()
    
def OLD_fig_energy_budget(E_CTR, E_FR, r_post_fit, fontsizes={'panelletterfontsize': 15}, figsize=(10,10), savename_mode=True):
    # create energy budget figure
    
    # input
    # E_CTR = [E_tot_CTR, E_HK_CTR, E_RP_CTR, E_AP_CTR, E_ST_CTR, E_glu_CTR, E_Ca_CTR]
    # E_FR  = [E_tot_FR,  E_HK_FR,  E_RP_FR,  E_AP_FR,  E_ST_FR,  E_glu_FR,  E_Ca_FR]
    # r_post_fit is array of firing rates (Hz)
    # fontsizes is a dictionary of used font sizes
    # figsize is the figure size
    # savename_mode decides whether fig is saved or not

    E_tot_CTR, E_HK_CTR, E_RP_CTR, E_AP_CTR, E_ST_CTR, E_glu_CTR, E_Ca_CTR = E_CTR
    E_tot_FR,  E_HK_FR,  E_RP_FR,  E_AP_FR,  E_ST_FR,  E_glu_FR,  E_Ca_FR  = E_FR

    r_post_optimum = 4.0  # Hz
    idx_optimum = (np.abs(r_post_fit - r_post_optimum)).argmin()

    description_CTR = 'CTR'
    description_FR  = 'FR'
    legend_pos = 'upper right'
    y_label = '$E_{tot}$ (ATP/s)'
    y_limit = 5.0e9

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1.0, 0.5, 0.5, 1.0], hspace=0.6, wspace=0.0)

    # top row: A spans cols 0–1, B spans cols 2–3
    axA = fig.add_subplot(gs[0, 0:2])
    axB = fig.add_subplot(gs[0, 2:4])

    # bottom row: C at col 0, D at col 3, middle cols are empty
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 3])
    
    """
    fig, axes = plt.subplots(2,4, figsize=figsize, gridspec_kw={'height_ratios': [1.0, 1.0], 'width_ratios': [1.0, 0.2, 0.2, 1.0], 'hspace': 0.6, 'wspace': 0.0})

    axA = axes[0, 0:2]
    axB = axes[0, 2:3]
    axC = axes[1, 0]
    axCspace = axes[1, 1]
    axDspace = axes[1, 2]
    axD = axes[1, 3]
    
    # switch off unused axes
    #axes[0, 0].axis('off')
    #axes[1, 0].axis('off')
    for j in range(2):
        axes[1, j].axis('off')"""
        
    labels=['$E_{HK}$', '$E_{RP}$', '$E_{AP}$', '$E_{glu}$', '$E_{Ca^{2+}}$', '$E_{syn}$']
    
    # panel A: CTR stackplot
    pf.plot_energy_stackplot(E_tot_CTR, E_HK_CTR, E_RP_CTR, E_AP_CTR, E_ST_CTR, E_glu_CTR, E_Ca_CTR, r_post_fit, description_CTR, labels, r_post_optimum, r_post_optimum_percentages=True, inverted=False, legend_pos=False, y_limit=y_limit, y_label=y_label, color_r_post_optimum=color_CTR, ax=axA)
    axA.set_title('CTR', fontweight='bold', color=color_CTR, fontsize=fontsizes['panelletterfontsize']*0.8)
    axA.text(-0.12, 1.04, "A", transform=axA.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    # panel B: FR stackplot (inverted x-axis, legend here)
    pf.plot_energy_stackplot(E_tot_FR, E_HK_FR, E_RP_FR, E_AP_FR, E_ST_FR, E_glu_FR, E_Ca_FR, r_post_fit, description_FR, labels, r_post_optimum, r_post_optimum_percentages=True, inverted=True, legend_pos=legend_pos, y_limit=y_limit, y_label=y_label, color_r_post_optimum=color_FR, ax=axB)
    axB.set_title('FR', fontweight='bold', color=color_FR, fontsize=fontsizes['panelletterfontsize']*0.8)
    #ppf.panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])
    axB.text(-0.01, 1.04, "B", transform=axB.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    # panel C: CTR pie
    pf.plot_energy_pie_chart(E_HK_CTR[idx_optimum], E_RP_CTR[idx_optimum], E_AP_CTR[idx_optimum], E_ST_CTR[idx_optimum], E_glu_CTR[idx_optimum], E_Ca_CTR[idx_optimum], r_post_optimum, title_mode=False, label_mode='long', ax=axC)
    axC.text(-0.19, 1.04, "C", transform=axC.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    # panel D: FR pie
    insetD = inset_axes(axD, width="90%", height="90%", loc="center")
    pf.plot_energy_pie_chart(E_HK_FR[idx_optimum], E_RP_FR[idx_optimum], E_AP_FR[idx_optimum], E_ST_FR[idx_optimum], E_glu_FR[idx_optimum], E_Ca_FR[idx_optimum], r_post_optimum, title_mode=False, label_mode='long', ax=insetD)
    for spine in axD.spines.values():
        spine.set_visible(False)
    axD.set_xticks([])
    axD.set_yticks([])
    axD.text(-0.5, 1.04, "D", transform=axD.transAxes, fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='bottom')
    
    
    # add vertical + horizontal connector lines A -- C (black)
    x_disp_A, _ = axA.transData.transform((r_post_optimum, 0.0))
    x_fig_A, _ = fig.transFigure.inverted().transform((x_disp_A, 0.0))
    posA = axA.get_position()
    posC = axC.get_position()
    y_top_A = posA.ymin
    y_bar_C = posC.ymax + 0.07
    fig.add_artist(Line2D([x_fig_A, x_fig_A], [y_top_A, y_bar_C], transform=fig.transFigure, color=color_CTR, linestyle='--', linewidth=2.0, zorder=3))
    fig.add_artist(Line2D([posC.xmin, posC.xmax], [y_bar_C, y_bar_C], transform=fig.transFigure, color=color_CTR, linestyle='-', linewidth=3.0, zorder=3))

    # add vertical + horizontal connector lines B -- D (red)
    x_disp_B, _ = axB.transData.transform((r_post_optimum, 0.0))
    x_fig_B, _ = fig.transFigure.inverted().transform((x_disp_B, 0.0))
    posB = axB.get_position()
    posD = axD.get_position()
    y_top_B = posB.ymin
    y_bar_D = posD.ymax + 0.07
    fig.add_artist(Line2D([x_fig_B, x_fig_B], [y_top_B, y_bar_D], transform=fig.transFigure, color=color_FR, linestyle='--', linewidth=2.0, zorder=3))
    fig.add_artist(Line2D([posD.xmin, posD.xmax], [y_bar_D, y_bar_D], transform=fig.transFigure, color=color_FR, linestyle='-', linewidth=3.0, zorder=3))

    if savename_mode is True:
        save_path = '../Figures/0_paper/fig_5_energy_budget.pdf'
        fig.savefig(save_path, bbox_inches='tight', transparent=True)
        print(f"Saved figure to {save_path}")
        
    plt.show()
    plt.close(fig)
    
    
# Old 3x3 Fig. 1
def OLD_fig_Pareto_optimality(figA, figB, figC, results_mean_spiking_trials, results_grid_point_to_exp_data=None, fontsizes={'panelletterfontsize': 15}, figsize=(9,7), savename_mode=True): 
    # create Pareto_optimality figure
    # input
    # figA, figB, figC are the 3D figures
    # results_mean_spiking_trials is a dictionary of the full grid results
    # results_grid_point_to_exp_data is a dictionary of the experimental data points fitted to their closest grid point 
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    fig = plt.figure(figsize=figsize, dpi=600)
    G = gs.GridSpec(4, 3, figure=fig, width_ratios=[1,1,1], height_ratios=[1,0.9,0.05,0.9], wspace=0.45, hspace=0.3)

    # A–C 
    axA=fig.add_subplot(G[0, 0])
    render_3Dfig_to_ax(figA, axA)
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    
    axB=fig.add_subplot(G[0, 1])
    render_3Dfig_to_ax(figB, axB)
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])
    
    axC=fig.add_subplot(G[0, 2])
    render_3Dfig_to_ax(figC, axC)
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    
    # smaller space
    #axspace=axes[3:6]
    
    # D -- I
    R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy = pf.load_data_correlations(results_mean_spiking_trials)

    # D
    axD=fig.add_subplot(G[1, 0])
    pf.plot_correlation(E_tot, MI, 'E_tot', 'MI', MI_per_energy, 'MI_per_energy', results_grid_point_to_exp_data=None, inverted_x=True, plot_mode='correlation_short', ax=axD)
    axD.xaxis.get_offset_text().set_y(2.0)
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    # E
    axE=fig.add_subplot(G[1, 1])
    pf.plot_correlation(E_tot, OSI, 'E_tot', 'OSI', OSI_per_energy, 'OSI_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, inverted_x=True, plot_mode='correlation_short', ax=axE)
    axE.xaxis.get_offset_text().set_y(20.0)
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'])

    # F
    axF=fig.add_subplot(G[1, 2])
    pf.plot_correlation(E_tot, TE, 'E_tot', 'TE', TE_per_energy, 'TE_per_energy', results_grid_point_to_exp_data=None, inverted_x=True, plot_mode='correlation_short', ax=axF)
    axF.xaxis.get_offset_text().set_y(0.0001)
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])

    # G: fit highlighted points of MI_per_energy
    axG=fig.add_subplot(G[3, 0])
    fit_func_highlight=pf.linear_rate_information_func # options: linear_func 2, sqrt_func 2, log_func 3, rate_information_func 3, linear_rate_information_func 5, sqrt_rate_information_func 5, linear_log_func 4, piecewise_linear_exponential_func 3
    initial_guess_highlight=[1,1,1,1,1]
    
    R_m_init, E_L_init = 150.0, -50.0
    x_label, y_label = 'r_post', 'MI_per_energy'
    w_scale_list_x, x_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, x_label, R_m_init, E_L_init)
    w_scale_list_y, y_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, y_label, R_m_init, E_L_init)
    highlight_points = (x_highlights, y_highlights)
    # z: E_tot, 'E_tot'
    pf.plot_correlation(r_post, MI_per_energy, x_label, y_label, highlight_points=highlight_points, fit_func_highlight=fit_func_highlight, initial_guess_highlight=initial_guess_highlight, plot_mode='correlation_short', ax=axG)
    axG.get_legend().remove()
    panel_letter(axG, "G", size=fontsizes['panelletterfontsize'])

    # H
    axF=fig.add_subplot(G[3, 1])
    pf.plot_correlation(r_post, MICE, 'r_post', 'MICE', MICE_per_energy, 'MICE_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, plot_mode='correlation_short', ax=axF)
    axF.set_xlim(right=25) 
    axF.get_legend().remove()
    panel_letter(axF, "H", size=fontsizes['panelletterfontsize'])

    # I (same correlation style but different layout/axis if desired)
    axI=fig.add_subplot(G[3, 2])
    pf.plot_correlation(r_post, MICE, 'r_post', 'MICE', MICE_per_energy, 'MICE_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, plot_mode='correlation_short', ax=axI)
    axI.get_legend().remove()
    panel_letter(axI, "I", size=fontsizes['panelletterfontsize'])

    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/0_paper/fig_1_Pareto_optimality.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)#, dpi=600)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)



def OLD_plot_ad_or_exp_tuning_curves(tuning_curves_CTR_LIF, tuning_curves_FR_LIF, tuning_curves_CTR_LIFad, tuning_curves_FR_LIFad, tuning_curves_CTR_LIFexp, tuning_curves_FR_LIFexp, tuning_curves_CTR_AdExp, tuning_curves_FR_AdExp,normalized=True, mean_over_zeros=False, mode='tuning_curve', titles=("LIF", "LIF+Ad", "LIF+Exp", "AdExp"), fontsizes={'panelletterfontsize': 15}, figsize=(7,2), savename_mode=None):
    # plot ad or exp tuning curves

    # input
    # tuning_curves_CTR, tuning_curves_FR are lists or arrays of the control & food-restricted tuning curves
    # label_CTR, label_FR are the labels of the plotted values
    # normalized is the option of normalizing the tuning curves
    # color_CTR, color_FR are the colors of the tuning curves
    # mean_over_zeros is an optional argument which excludes all 0 values before meaning
    # mode decides if tuning_curve labels or CV_ISI_labels are used
    # titles are the titles for the subfigures
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode determines if the figures is saved or not
    
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, 4, figure=fig, wspace=0.25, hspace=0.0)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # A: LIF (y-label shown)
    axA = axes[0]
    pf.plot_tuning_curves(tuning_curves_CTR_LIF, tuning_curves_FR_LIF, label_CTR='CTR', label_FR='FR', normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=False, show_xlabel=True, show_ylabel=True, ax=axA)
    axA.set_title(titles[0])
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    #axA.text(-0.12, 1.03, "A", transform=axA.transAxes, fontweight='bold', va='bottom', ha='left') 

    # B: LIF+ad (no y-label)
    axB = axes[1]
    pf.plot_tuning_curves(tuning_curves_CTR_LIFad, tuning_curves_FR_LIFad, label_CTR='CTR', label_FR='FR', normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=False, show_xlabel=True, show_ylabel=False, ax=axB)
    axB.set_title(titles[1])
    axB.set_yticklabels([])
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])
    #axB.text(-0.12, 1.03, "B", transform=axB.transAxes, fontweight='bold', va='bottom', ha='left')

    # C: LIF+exp (no y-label)
    axC = axes[2]
    pf.plot_tuning_curves(tuning_curves_CTR_LIFexp, tuning_curves_FR_LIFexp, label_CTR='CTR', label_FR='FR', normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=False, show_xlabel=True, show_ylabel=False, ax=axC)
    axC.set_title(titles[2])
    axC.set_yticklabels([])
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    #axC.text(-0.12, 1.03, "C", transform=axC.transAxes, fontweight='bold', va='bottom', ha='left')

    # D: AdExp (legend only here; no y-label)
    axD = axes[3]
    pf.plot_tuning_curves(tuning_curves_CTR_AdExp, tuning_curves_FR_AdExp, label_CTR='CTR', label_FR='FR', normalized=normalized, mean_over_zeros=mean_over_zeros, mode=mode, show_legend=True, show_xlabel=True, show_ylabel=False, ax=axD)
    axD.set_title(titles[3])
    axD.set_yticklabels([])
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])
    #axD.text(-0.12, 1.03, "D", transform=axD.transAxes, fontweight='bold', va='bottom', ha='left') 
    
    if savename_mode is True:
        #plt.tight_layout()
        savepath='../Figures/0_paper/fig_S3_ad_or_exp.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        plt.show()
        
# Fig. 1
def fig_Pareto_optimality_Padamsey_plotly(figA, figB, figC, results_mean_spiking_trials, results_grid_point_to_exp_data=None, colors=['black', 'red', 'yellow'], fontsizes={'panelletterfontsize': 15}, figsize=(9,4.5), savename_mode=True): 
    # create Pareto_optimality figure
    # input
    # figA, figB, figC are the 3D figures
    # results_mean_spiking_trials is a dictionary of the full grid results
    # results_grid_point_to_exp_data is a dictionary of the experimental data points fitted to their closest grid point 
    # colors are the colors for CTR, FR & highlight points
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    #color_CTR, color_FR, color_highlight = colors[0], colors[1], colors[2]
    
    fig = plt.figure(figsize=figsize, dpi=300) # 600
    G = gs.GridSpec(2, 3, figure=fig, width_ratios=[1,1,1], height_ratios=[1.2,1], wspace=0.45, hspace=0.3)

    # A–C 
    axA=fig.add_subplot(G[0, 0])
    render_3Dfig_to_ax(figA, axA)
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    
    axB=fig.add_subplot(G[0, 1])
    render_3Dfig_to_ax(figB, axB)
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])
    
    axC=fig.add_subplot(G[0, 2])
    render_3Dfig_to_ax(figC, axC)
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    
    # smaller space
    #axspace=axes[3:6]
    
    # D -- I
    R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy, MI_new, MI_new_per_energy, MICE_new, MICE_new_per_energy, TE_new, TE_new_per_energy, TECE_new, TECE_new_per_energy  = pf.load_data_correlations(results_mean_spiking_trials)
    #R_m, E_L, w_scale, I_syn_e, r_post, E_tot, CV_V_m, OSI, OSI_per_energy, MI_tuning_curve, MI_tuning_curve_per_energy, MICE_tuning_curve, MICE_tuning_curve_per_energy, MI_post, MI_post_per_energy, MICE_post, MICE_post_per_energy, CV_ISI, CV_ISI_per_energy, MI, MI_per_energy, TE, TE_per_energy, MICE, MICE_per_energy, TECE, TECE_per_energy = pf.load_data_correlations(results_mean_spiking_trials)

    # D
    axD=fig.add_subplot(G[1, 0])
    pf.plot_correlation(E_tot, OSI, 'E_tot', 'OSI', OSI_per_energy, 'OSI_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, inverted_x=True, colors=colors, plot_mode='correlation_short', ax=axD)
    axD.xaxis.get_offset_text().set_y(20.0)
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])

    # E
    axE=fig.add_subplot(G[1, 1])
    pf.plot_correlation(r_post, MICE, 'r_post', 'MICE', MICE_per_energy, 'MICE_per_energy', results_grid_point_to_exp_data=results_grid_point_to_exp_data, colors=colors, plot_mode='correlation_short', ax=axE)
    axE.set_xlim(left=-0.5, right=10) 
    axE.get_legend().remove()
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'])

    # F
    axF=fig.add_subplot(G[1, 2])
    #fit_func_highlight=pf.linear_rate_information_func # options: linear_func 2, sqrt_func 2, log_func 3, rate_information_func 3, linear_rate_information_func 5, sqrt_rate_information_func 5, linear_log_func 4, piecewise_linear_exponential_func 3
    #initial_guess_highlight=[1,1,1,1,1]
    
    #R_m_init, E_L_init = 150.0, -50.0
    #x_label, y_label = 'r_post', 'MI_per_energy'
    
    #w_scale_list_x, x_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, x_label, R_m_init, E_L_init)
    #w_scale_list_y, y_highlights = pf.get_w_scale_value_data(results_mean_spiking_trials, y_label, R_m_init, E_L_init)
        
    #highlight_points = (x_highlights, y_highlights)
    
    # z: E_tot, 'E_tot'
    #pf.plot_correlation(r_post, MI_per_energy, x_label, y_label, highlight_points=highlight_points, fit_func_highlight=fit_func_highlight, initial_guess_highlight=initial_guess_highlight, colors=colors, plot_mode='correlation_short', ax=axF)
    pf.plot_correlation(r_post, MI_per_energy, 'r_post', 'MI_per_energy', E_tot, 'E_tot', results_grid_point_to_exp_data=results_grid_point_to_exp_data, colors=colors, plot_mode='correlation_short', ax=axF)
    axF.get_legend().remove()
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])

    plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_1_Pareto_optimality_Padamsey.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)#, dpi=600)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)


#Fig. 2
def fig_Pareto_optimality_Zeldenrust_plotly(figA, figB, results_analysis_exc, results_analysis_inh, results_PC_exc, results_PC_inh, proportion_of_synaptic_change, MI_list, E_tot_list, colors=['coral', 'cornflowerblue', 'red'], fontsizes={'panelletterfontsize': 15}, figsize=(9,9), savename_mode=True): 
    # create Pareto optimality figure for Zeldenrust data
    # input
    # figA, figB, figC are the 3D figures
    # results_analysis_exc & results_analysis_inh are dictionaries of the experimental data points 
    # results_PC_exc, results_PC_inh are dictionaries of the PC grids
    # proportion_of_synaptic_change, MI_list, E_tot_list are the lists of predicted FR trajectories for Zeldenrust data
    # colors are the colors for excitatory & inhibitory plots & color_FR
    # fontsizes is a dictionary of used font sizes
    # figsize determines the figsize
    # savename_mode decides whether the figure is saved or not
    
    color_exc, color_inh, color_FR = colors[0], colors[1], colors[2]
    fig = plt.figure(figsize=figsize, dpi=600)
    G = gs.GridSpec(3, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 1], height_ratios=[1.3, 1.0, 1.0], wspace=1.0, hspace=0.5)

    # A
    axA = fig.add_subplot(G[0, 0:3])
    render_3Dfig_to_ax(figA, axA)
    panel_letter(axA, "A", size=fontsizes['panelletterfontsize'])
    
    # B
    axB = fig.add_subplot(G[0, 3:6])
    render_3Dfig_to_ax(figB, axB)
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])
    
    # C: excitatory Pareto
    row2 = G[1, :].subgridspec(1, 3, wspace=0.5) # extra width space

    axC = fig.add_subplot(row2[0, 0])
    #axC = fig.add_subplot(G[1, 0:2])
    pc1_grid_exc, pc2_grid_exc, pc3_grid_exc, E_L_grid_exc, V_thresh_grid_exc, R_m_grid_exc, C_m_grid_exc, Delta_T_ad_grid_exc, V_reset_grid_exc, tau_w_ad_grid_exc, a_ad_grid_exc, b_ad_grid_exc, V_m_grid_exc, I_syn_e_grid_exc, I_syn_i_grid_exc, r_post_grid_exc, E_tot_grid_exc, CV_V_m_grid_exc, CV_ISI_grid_exc, CV_ISI_per_energy_grid_exc, MI_grid_exc, MI_per_energy_grid_exc, MICE_grid_exc, MICE_per_energy_grid_exc, hit_fraction_grid_exc, false_alarm_fraction_grid_exc = pf.load_data_correlations_Zeldenrust(results_PC_exc, r_post_upper_bound=100)
    
    x_exp_name = 'E_tot_1e9_ATP_per_s_list'
    y_exp_name = 'MI_FZ_bits_list' 
    #z_exp_name = 'MI_FZ_per_energy_list' 
    
    x_grid_exc = np.asarray(E_tot_grid_exc)/1e9
    y_grid_exc = np.asarray(MI_grid_exc)
    z_grid_exc = np.asarray(MI_per_energy_grid_exc)
    
    x_min_grid_exc = min(x_grid_exc)
    y_max_grid_exc = max(y_grid_exc)
    z_max_grid_exc = max(z_grid_exc)
    
    pf.plot_correlation(x_grid_exc/x_min_grid_exc, y_grid_exc/y_max_grid_exc, 'normalized $E_{tot}$', 'normalized $MI$', z=z_grid_exc/z_max_grid_exc, z_label='norm. $MI$ per energy', ax=axC) #, inverted_x=True
    
    x_exp_exc = np.asarray(results_analysis_exc[x_exp_name])
    y_exp_exc = results_analysis_exc[y_exp_name]
    #z_exp_exc = results_analysis_exc[z_exp_name]
    
    x_min_exp_exc = min(np.asarray(x_exp_exc))
    y_max_exp_exc = max(np.asarray(y_exp_exc))
    #z_max_exp_exc = max(np.asarray(z_exp_exc))

    pf.plot_correlation_exc_inh(x_exp_exc/x_min_exp_exc, [], y_exp_exc/y_max_exp_exc, [], 'normalized $E_{tot}$', 'normalized $MI$', inverted_x=True, colors=[color_exc, color_inh], transparency=0.7, ax=axC)

    axC.get_legend().remove()
    """
    axC.set_xscale('log')
    xmin, xmax = axC.get_xlim()
    ticks = 10 ** np.linspace(np.log10(xmin), np.log10(xmax),3)
    from matplotlib.ticker import LogFormatterMathtext, NullLocator, NullFormatter
    axC.set_xticks(ticks)
    axC.xaxis.set_major_formatter(LogFormatterMathtext())

    # kill minor ticks + labels completely
    axC.xaxis.set_minor_locator(NullLocator())
    axC.xaxis.set_minor_formatter(NullFormatter())
    """
    """
    handles, labels = axC.get_legend_handles_labels()
    exc_handles = [h for h, l in zip(handles, labels) if "exc" in l.lower()]
    exc_labels  = [l for l in labels if "exc" in l.lower()]
    axC.legend(exc_handles, exc_labels, frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', loc='upper left')
    """
    
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])
    
    # D: inhibitory Pareto
    axD = fig.add_subplot(row2[0, 1])
    #axD = fig.add_subplot(G[1, 2:4])
    pc1_grid_inh, pc2_grid_inh, pc3_grid_inh, E_L_grid_inh, V_thresh_grid_inh, R_m_grid_inh, C_m_grid_inh, Delta_T_ad_grid_inh, V_reset_grid_inh, tau_w_ad_grid_inh, a_ad_grid_inh, b_ad_grid_inh, V_m_grid_inh, I_syn_e_grid_inh, I_syn_i_grid_inh, r_post_grid_inh, E_tot_grid_inh, CV_V_m_grid_inh, CV_ISI_grid_inh, CV_ISI_per_energy_grid_inh, MI_grid_inh, MI_per_energy_grid_inh, MICE_grid_inh, MICE_per_energy_grid_inh, hit_fraction_grid_inh, false_alarm_fraction_grid_inh = pf.load_data_correlations_Zeldenrust(results_PC_inh, r_post_upper_bound=100)

    x_grid_inh = np.asarray(E_tot_grid_inh)/1e9
    y_grid_inh = np.asarray(MI_grid_inh)
    z_grid_inh = np.nan_to_num(np.asarray(MI_per_energy_grid_inh), nan=0.0)

    x_min_grid_inh =  min(x_grid_inh)
    y_max_grid_inh = max(y_grid_inh)
    z_max_grid_inh = max(z_grid_inh)

    pf.plot_correlation(x_grid_inh/x_min_grid_inh, y_grid_inh/y_max_grid_inh, 'normalized $E_{tot}$', 'normalized $MI$', z=z_grid_inh/z_max_grid_inh, z_label='norm. $MI$ per energy', ax=axD)  #, inverted_x=True

    x_exp_inh = np.asarray(results_analysis_inh[x_exp_name])
    y_exp_inh = results_analysis_inh[y_exp_name]
    #z_exp_inh = results_analysis_inh[z_exp_name]

    x_min_exp_inh = min(np.asarray(x_exp_inh))
    y_max_exp_inh = max(np.asarray(y_exp_inh))
    #z_max_exp_inh = max(np.asarray(z_exp_inh))

    pf.plot_correlation_exc_inh([], x_exp_inh/x_min_exp_inh, [], y_exp_inh/y_max_exp_inh, 'normalized $E_{tot}$', 'normalized $MI$', inverted_x=True, colors=[color_exc, color_inh], transparency=0.7, ax=axD)

    axD.get_legend().remove()
    """
    handles, labels = axD.get_legend_handles_labels()
    inh_handles = [h for h, l in zip(handles, labels) if "inh" in l.lower()]
    inh_labels  = [l for l in labels if "inh" in l.lower()]
    axD.legend(inh_handles, inh_labels, frameon=True, facecolor='white', framealpha=1.0, edgecolor='none', loc='lower left')
    """
    panel_letter(axD, "D", size=fontsizes['panelletterfontsize'])
    
    # E
    axE = fig.add_subplot(row2[0, 2])
    #axE = fig.add_subplot(G[1, 4:6])

    x_name = 'firing_rate_calculated_Hz_windowed'
    y_name_MICE_FZ = 'MICE_FZ_windowed_list'
    firing_rate_Hz_windowed_list_flat_exc = np.concatenate(results_analysis_exc[x_name]).tolist()
    MICE_FZ_bits_windowed_list_flat_exc = np.concatenate(results_analysis_exc[y_name_MICE_FZ]).tolist()
    firing_rate_Hz_windowed_list_flat_inh = np.concatenate(results_analysis_inh[x_name]).tolist()
    MICE_FZ_bits_windowed_list_flat_inh = np.concatenate(results_analysis_inh[y_name_MICE_FZ]).tolist()
    pf.plot_correlation_exc_inh(firing_rate_Hz_windowed_list_flat_exc, firing_rate_Hz_windowed_list_flat_inh, np.array(MICE_FZ_bits_windowed_list_flat_exc)*3*4, np.array(MICE_FZ_bits_windowed_list_flat_inh)*18*4, '$r_{post}$ (Hz)', '$CE_{MI}$ (bits/Hz)', colors=[color_exc, color_inh], ax=axE)
    panel_letter(axE, "E", size=fontsizes['panelletterfontsize'])
    
    # F
    row3 = G[2, :].subgridspec(1, 3, wspace=0.5) # extra width space
    
    axF = fig.add_subplot(row3[0, 0])
    x_name = 'R_m_mean_MOhm_list'
    y_name = 'tau_w_mean_ms_list'
    pf.plot_correlation_exc_inh(results_analysis_exc[x_name], results_analysis_inh[x_name], results_analysis_exc[y_name], results_analysis_inh[y_name], x_name, y_name, colors=[color_exc, color_inh], ax=axF)
    axF.get_legend().remove()
    panel_letter(axF, "F", size=fontsizes['panelletterfontsize'])
    
    # G
    axG = fig.add_subplot(row3[0, 1])#fig.add_subplot(G[2, 1:3])
    x_name = 'R_m_mean_MOhm_list'
    y_name = 'E_L_mV_list' 
    
    pf.plot_correlation_exc_inh(results_analysis_exc[x_name], results_analysis_inh[x_name], results_analysis_exc[y_name], results_analysis_inh[y_name], x_name, y_name, colors=[color_exc, color_inh], ax=axG)
    pf.add_banana_strip(axG, x0=220, x1=350, y_top=-58, y_bottom=-70, power=3.8, lw=30, color=color_FR, alpha=0.4, zorder=10)
    if axG.get_legend() is not None:
        axG.get_legend().remove()
    
    panel_letter(axG, "G", size=fontsizes['panelletterfontsize'])
    
    # H: MI & E_tot FR
    hcell = row3[0, 2].subgridspec(2, 1, hspace=0.25, height_ratios=[1, 1])

    axH1 = fig.add_subplot(hcell[0, 0])  # MI
    axH2 = fig.add_subplot(hcell[1, 0])  # E_tot

    pf.plot_gradient_line(axH1, proportion_of_synaptic_change, MI_list, lw=4, alpha_left=0.3, alpha_right=1.0, color=color_FR, zorder=5)
    axH1.set_ylabel('$MI$ (bits)')
    axH1.set_xlabel("")
    axH1.tick_params(axis="x", labelbottom=False)
    axH1.spines['top'].set_visible(False)
    axH1.spines['right'].set_visible(False)

    pf.plot_gradient_line(axH2, proportion_of_synaptic_change, np.asarray(E_tot_list) / 1e9, lw=4, alpha_left=0.3, alpha_right=1.0, color=color_FR, zorder=5)
    axH2.set_xlabel('Prop. of FR syn. weight change')
    axH2.set_ylabel('$E_{tot}$ ($10^9$ATP/s)')
    axH2.spines['top'].set_visible(False)
    axH2.spines['right'].set_visible(False)

    # Panel letter für G (oben links im oberen Subplot)
    panel_letter(axH1, "H", size=fontsizes['panelletterfontsize'])
    
    #plt.tight_layout()
    if savename_mode is True:
        savepath="../Figures/0_paper/fig_2_Pareto_optimality_Zeldenrust.pdf"
        plt.savefig(savepath, bbox_inches='tight', transparent=True)#, dpi=600)
        plt.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
        
    print('norm. const. $E_{tot}$ grid exc: ' + str(round(x_min_grid_inh,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ grid exc: ' + str(round(y_max_grid_inh,4)) + ' bits')
    print('norm. const. $MI$ per energy grid exc: ' + str(round(z_max_grid_exc,4)) + ' bits/($10^{9}$ ATP/s)') 
    
    print('norm. const. $E_{tot}$ exp exc: ' + str(round(x_min_exp_exc,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ exp exc: ' + str(round(y_max_exp_exc,4)) + ' bits') 
    #print('norm. const. $MI$ per energy exp exc: ' + str(round(z_max_exp_exc,4)) + ' bits/($10^{9}$ ATP/s)') 
    
    print('norm. const. $E_{tot}$ grid inh: ' + str(round(x_min_grid_inh,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ grid inh: ' + str(round(y_max_grid_inh,4)) + ' bits')  
    print('norm. const. $MI$ per energy grid exc: ' + str(round(z_max_grid_inh,4)) + ' bits/($10^{9}$ ATP/s)') 
    
    print('norm. const. $E_{tot}$ exp inh: ' + str(round(x_min_exp_inh,4)) + ' $10^{9}$ ATP/s')
    print('norm. const. $MI$ grid exc: ' + str(round(y_max_exp_inh,4)) + ' bits')
    #print('norm. const. $MI$ per energy exp exc: ' + str(round(z_max_exp_inh,4)) + ' bits/($10^{9}$ ATP/s)') 
        

def fig_multiple_trajectories_examples_plotly(fig3d, data_CTR_like, data_FR_like, colors=['#4a61e8', '#c41c5f', 'black'], fontsizes={'panelletterfontsize': 15}, figsize=(5,8), savename_mode=True):
    # compose the 3D + combined example traces + histogram figure
    #
    # input
    # fig3d is the 3D figure (with trajectories)
    # data_CTR_like, data_FR_like are dictionaries of the CTR_like & FR_like points
    # colors are the colors for color_CTR_like, color_FR_like, color_CTR
    # fontsizes is a dictionary of used font sizes
    # figsize is the figure size
    # savename_mode decides whether fig is saved or not
    
    color_CTR_like, color_FR_like, color_CTR = colors[0], colors[1], colors[2]
    # extract CTR-like and FR-like locations and convert w_scale → w_e
    R_m_CTR_like, E_L_CTR_like, w_scale_CTR_like = data_CTR_like['R_m'], data_CTR_like['E_L'], data_CTR_like['w_scale']
    R_m_FR_like, E_L_FR_like, w_scale_FR_like = data_FR_like['R_m'], data_FR_like['E_L'], data_FR_like['w_scale']

    w_e_CTR_like = float(pf.w_scale_to_w_e_syn(w_scale_CTR_like))
    w_e_FR_like = float(pf.w_scale_to_w_e_syn(w_scale_FR_like))

    # add CTR-like & FR-like markers to the 3D fig
    fig3d.add_trace(go.Scatter3d(x=[R_m_CTR_like], y=[E_L_CTR_like], z=[w_e_CTR_like], mode='markers', marker=dict(size=15, color=color_CTR_like, line=dict(width=1, color='white')), showlegend=False))
    fig3d.add_trace(go.Scatter3d(x=[R_m_FR_like], y=[E_L_FR_like], z=[w_e_FR_like], mode='markers', marker=dict(size=15, color=color_FR_like, line=dict(width=1, color='white')), showlegend=False))
    fig3d.update_layout(showlegend=False)

    # save 3D fig as PNG to embed in final figure
    three_d_png_path = '../Figures/0_paper/fig_multiple_trajectory_examples_3d.png'
    pio.write_image(fig3d, three_d_png_path, format='png', scale=2)

    # simulated V_m traces
    V_m_CTR_like = data_CTR_like['V_m']
    V_m_FR_like = data_FR_like['V_m']

    # shift V_m so resting level is at 0 mV
    V_m_shifted_CTR_like, _ = pf.shift_V_m(V_m_CTR_like, target_mV=0.0)
    V_m_shifted_FR_like, _ = pf.shift_V_m(V_m_FR_like, target_mV=0.0)

    # cut excerpts
    V_m_excerpts_CTR_like, V_m_min_CTR_like, V_m_max_CTR_like, _ = pf.cut_into_excerpts(V_m_shifted_CTR_like, n_excerpts=9, start_idx=3000, stop_idx=30000)
    V_m_excerpts_FR_like, V_m_min_FR_like, V_m_max_FR_like, _ = pf.cut_into_excerpts(V_m_shifted_FR_like, n_excerpts=9, start_idx=3000, stop_idx=30000)

    # shared y-limits
    V_m_min, V_m_max = pf.compute_shared_ylim(V_m_min_CTR_like, V_m_max_CTR_like, V_m_min_FR_like, V_m_max_FR_like)

    # create final figure (2 rows × 2 columns; A spans full top row)
    fig = plt.figure(figsize=figsize)
    G = gs.GridSpec(2, 2, figure=fig, height_ratios=[4.5, 1.5], wspace=0.20, hspace=0.08)
    fig.subplots_adjust(top=0.98, bottom=0.14, left=0.06, right=0.99)

    # Panel A: 3D figure (spanning full upper row)
    ax_top = fig.add_subplot(G[0, :])
    im = Image.open(three_d_png_path).convert("RGBA")
    im_cropped = im.crop(im.getbbox())
    ax_top.imshow(np.array(im_cropped))
    ax_top.axis('off')
    top_bbox = ax_top.get_position()
    fig.text(top_bbox.x0 - 0.01, top_bbox.y1 - 0.15, 'A', fontsize=fontsizes['panelletterfontsize'], fontweight='bold', ha='left', va='top')
    #ppf.panel_letter(ax_top, "A", size=fontsizes['panelletterfontsize'])

    # B: combined excerpts
    axB = fig.add_subplot(G[1, 0])
    pf.plot_two_V_m_excerpts(V_m_excerpts_1=V_m_excerpts_CTR_like, V_m_excerpts_2=V_m_excerpts_FR_like, color_1=color_CTR_like, color_2=color_FR_like, ylims=(V_m_min, V_m_max), ax=axB)
    panel_letter(axB, "B", size=fontsizes['panelletterfontsize'])

    # add colored parameter text below panel B
    Bbox = axB.get_position()
    x_center = (Bbox.x0 + Bbox.x1) / 1.9
    y_base = Bbox.y0 - 0.05
    dy = 0.025
    x_label = x_center - 0.1
    x_CTR_like = x_center + 0.13
    x_FR_like = x_center + 0.01
    
    # R_m line
    fig.text(x_label, y_base + 2*dy, r'$R_m$:', ha='left', va='top')
    fig.text(x_FR_like, y_base + 2*dy, rf'${R_m_FR_like:.0f}\,\mathrm{{M\Omega}}$', ha='left', va='top', color=color_FR_like)
    fig.text(x_CTR_like, y_base + 2*dy, rf'${R_m_CTR_like:.0f}\,\mathrm{{M\Omega}}$', ha='left', va='top', color=color_CTR_like)
    
    # V_rest line
    fig.text(x_label, y_base + dy, r'$V_{\mathrm{rest}}$:', ha='left', va='top')
    fig.text(x_FR_like, y_base + dy, rf'${E_L_FR_like:.0f}\,\mathrm{{mV}}$', ha='left', va='top', color=color_FR_like)
    fig.text(x_CTR_like, y_base + dy, rf'${E_L_CTR_like:.0f}\,\mathrm{{mV}}$', ha='left', va='top', color=color_CTR_like)
    
    # <w_syn,e> line
    fig.text(x_label, y_base + 0*dy, r'$\langle w_{\mathrm{syn,e}}\rangle$:', ha='left', va='top')
    fig.text(x_FR_like, y_base + 0*dy, rf'${w_e_FR_like:.2f}\,\mathrm{{nS}}$', ha='left', va='top', color=color_FR_like)
    fig.text(x_CTR_like, y_base + 0*dy, rf'${w_e_CTR_like:.2f}\,\mathrm{{nS}}$', ha='left', va='top', color=color_CTR_like)
    
    # C: histogram
    xmin = -45
    xmax = -80
    V_thresh = data_CTR_like['V_thresh']

    axC = fig.add_subplot(G[1, 1])
    axC.hist(V_m_FR_like, bins=np.linspace(xmax, xmin, 80), color=color_FR_like, alpha=0.8, density=True)
    axC.hist(V_m_CTR_like, bins=np.linspace(xmax, xmin, 80), color=color_CTR_like, alpha=0.8, density=True)
    axC.axvline(V_thresh, 0, 0.85, color=color_CTR, ls='--', alpha=0.9)
    axC.text(V_thresh, 0.27, r'$V_{\mathrm{thresh}}$', ha='center', va='bottom')
    axC.set_xlim(xmin, xmax)
    axC.set_ylim(0, 0.3)
    axC.set_xlabel(r'$V_m$ (mV)')
    axC.yaxis.set_visible(False)
    axC.spines['left'].set_visible(False)
    axC.spines['top'].set_visible(False)
    axC.spines['right'].set_visible(False)
    axC.xaxis.set_inverted(True)
    axC.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    panel_letter(axC, "C", size=fontsizes['panelletterfontsize'])

    # save & show
    if savename_mode:
        plt.tight_layout()
        savepath = '../Figures/0_paper/fig_4_multiple_trajectory_examples.pdf'
        fig.savefig(savepath, bbox_inches='tight', transparent=True)
        fig.savefig(savepath.replace('.pdf', '.png'), dpi=120, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close(fig)
    
    
def panel_letter_2D(ax, letter, size=15):
    # write letter on panel
    # input
    # ax is the given axis
    # letter is the desired letter to show
    
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=size, fontweight='bold', ha='left', va='bottom')

