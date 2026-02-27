######################## import packages ########################
import Functions.analysis_functions as af
import Functions.plotting_functions as pf

import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from scipy.io import loadmat
from scipy.signal import decimate
import seaborn as sns
import time as tp

import pickle as pkl
import pandas as pd

import torch
from sbi import utils as sbi_utils
from sbi import inference as sbi_inference
#from sbi.neural_nets import posterior_nn
from functools import partial
import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

#import sys
#sys.path.append('/Users/psommer/Downloads/Data_Zeldenrust_2024/Scripts')

#from pathlib import Path
#LOG_DIR = Path("/Users/psommer/Downloads/Data_Zeldenrust_2024/Data_SBI_logs/sequential_large")
#LOG_DIR.mkdir(parents=True, exist_ok=True)

from pathlib import Path
import os

DATA_ROOT = Path(os.environ.get("ZELDENRUST_DATA_ROOT", "/Users/psommer/Downloads/Data_Zeldenrust_2024")).resolve()
LOG_DIR = DATA_ROOT / "Data_SBI_logs" / "sequential_large"

######################## load functions ########################

def load_data_experiment_Zeldenrust(data):
    # extract data from matlab file
    # input 
    # data is a matlab data file

    # output
    # results is a dictionary with the desired data
    
    # extract input generation settings
    input_generation_settings = getattr(data, 'input_generation_settings', None)
    cell_type = getattr(input_generation_settings, 'cell_type', None)
    tau_switching = getattr(input_generation_settings, 'tau', None)
    duration = getattr(input_generation_settings, 'duration', None)
    sampling_rate = getattr(input_generation_settings, 'sampling_rate', None)
    condition = getattr(input_generation_settings, 'condition', None)

    # extract general data
    hidden_state = getattr(data, 'hidden_state', None)
    input_current = getattr(data, 'input_current', None)
    membrane_potential = getattr(data, 'membrane_potential', None)
    mean_threshold = getattr(data, 'mean_threshold', None)
    firing_rate = getattr(data, 'firing_rate', None)
    spike_indices = getattr(data, 'spikeindices', None)

    # extract Analysis data
    Analysis = getattr(data, 'Analysis', None)  
    
    MI = [getattr(entry, 'MI', None) for entry in Analysis]
    FI = [getattr(entry, 'FI', None) for entry in Analysis]
    nup = [getattr(entry, 'nup', None) for entry in Analysis]
    ndown = [getattr(entry, 'ndown', None) for entry in Analysis]
    nspikeperup = [getattr(entry, 'nspikeperup', None) for entry in Analysis]
    nspikeperdown = [getattr(entry, 'nspikeperdown', None) for entry in Analysis]
    
    
    """
    # old version only suited for excitatory cells
    Analysis_0 = Analysis[0]
    Analysis_1 = Analysis[1]
    Analysis_2 = Analysis[2]

    MI = [getattr(Analysis_0, 'MI', None),
          getattr(Analysis_1, 'MI', None),
          getattr(Analysis_2, 'MI', None)]

    FI = [getattr(Analysis_0, 'FI', None),
          getattr(Analysis_1, 'FI', None),
          getattr(Analysis_2, 'FI', None)]
    """
    
    # return all values in a dictionary
    results = { 'cell_type': cell_type,
                'tau_switching_ms': tau_switching,
                'duration_ms': duration,
                'sampling_rate_per_ms': sampling_rate,
                'condition': condition,
                'hidden_state': hidden_state,
                'input_current_pA': input_current,
                'membrane_voltage_mV': membrane_potential,
                'membrane_threshold_mV': mean_threshold,
                'firing_rate_Hz': firing_rate,
                'spike_indices': spike_indices,
                'spike_times_ms': spike_indices/sampling_rate,
                'MI_bits': MI,
                'FI': FI, 
                'nup': nup,
                'ndown': ndown,
                'nspikeperup': nspikeperup,
                'nspikeperdown': nspikeperdown}

    return results


######################## estimate R_m & E_L function ########################

def estimate_R_m_E_L_subthreshold(V_m, I_inj, V_thresh, printing_mode=False, plotting_mode=False):
    # estimate membrane resistance R_m from linear regression of subthreshold data (V_m < mean threshold)
    # input
    # V_m is an array of the membrane potential trace in mV
    # I_inj is an array of the injected current trace in pA
    # V_thresh is an array of threshold voltages in mV
    # printing_modeis an optional argument whether to print the results
    # plotting_mode is an optional argument whether to generate a sub-/suprathreshold IV plot

    # output
    # R_m is the estimated resistance in MΩ
    # E_L is the estimated resting potential in mV
    # R² is the coefficient of determination
    # slope, intercept are fit parameters

    # thresholding out spikes
    V_thresh_mean = np.mean(V_thresh)
    mask_sub = V_m < V_thresh_mean
    mask_sup = ~mask_sub

    # linear regression (convert pA → nA for MΩ)
    slope, intercept, r_value, p_value, std_err = linregress(I_inj[mask_sub] / 1000, V_m[mask_sub])
    R_m = slope  # mV/nA = MΩ
    E_L = intercept
    R2 = r_value ** 2
    
    if printing_mode is True: 
        print(f"Estimated $R_m$ (subthreshold only): {R_m:.2f} MOhm (R² = {R2:.3f})")
        print(f"Estimated $E_L$ (subthreshold only): {E_L:.2f} mV (R² = {R2:.3f})")

    if plotting_mode:
        # fit line
        x_fit = np.linspace(np.min(I_inj[mask_sub]) / 1000, np.max(I_inj[mask_sub]) / 1000, 100)
        y_fit = slope * x_fit + intercept

        plt.figure(figsize=(8, 5))
        plt.scatter(I_inj[mask_sub] / 1000, V_m[mask_sub], color='orange', s=5, label=r'$V_m < V_{\mathrm{thresh}}$')
        plt.scatter(I_inj[mask_sup] / 1000, V_m[mask_sup], color='steelblue', s=5, alpha=0.5, label=r'$V_m \ge V_{\mathrm{thresh}}$')
        plt.plot(x_fit, y_fit, 'k--', label=fr'Fit: $R_m={slope:.2f}\,\mathrm{{MOhm}},\ E_L={intercept:.2f}\,\mathrm{{mV}},\ R^2={R2:.3f}$')

        plt.xlabel(r'Injected current / nA')
        plt.ylabel(r'Membrane voltage / mV')
        plt.title('IV Curve: Subthreshold vs. Suprathreshold')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return R_m, E_L, R2


######################## SBI preparatory functions ########################

def downsample_decimate(x, sampling_rate):
    # simple decimate-based downsampling
    # input
    # x is a 1D array
    # sampling_rate is the (integer) sampling rate of the experimental data

    # output
    # x_downsampled is a 1D array of the downsampled x
    
    x_downsampled = decimate(x, int(sampling_rate*0.1), ftype='fir', zero_phase=True) # with 0.1 being the target sampling rate for x_downsampled in ms
    return x_downsampled

def extract_max_spikes_window(t_vec_exp, V_m_exp, I_inj_exp, spike_times_post_exp, dt_original=0.1, T_window=2000, early_return_rate=15):
    # extracts a time window with the most spikes from experimental voltage
    
    # input
    # t_vec_exp is the experimental time vector in ms
    # V_m_exp is the experimentally measured voltage trace in mV
    # I_inj_exp is the experimentally injected current in pA
    # spike_times_post_exp is a list of the experimental spike times in ms
    # dt_original is the experimental sampling rate in ms
    # T_window is the length of the extracted time window in ms
    # early_return_rate is the maximum spiking rate to be in a window in Hz

    # output
    # t_vec_window is the windowed time vector in ms
    # V_m_exp_window is the windowed experimentally measured voltage trace in mV
    # I_inj_exp_window is the windowed experimentally injected current in pA
    # spike_times_post_exp_window are the windowed experimental spike times in ms
    # start_idx/best_start_idx is is the start index of the extracted time window    
    # end_idx is is the end index of the extracted time window

    T_window_sim_steps = int(round(T_window/dt_original)) # number of samples in 1 window
    
    total_steps = len(t_vec_exp) # get total number of time steps

    early_return_spikes = early_return_rate * T_window / 1000 # translate the desired max rate into the number of spikes in a window to be returned

    # initialize max spiking number & respective idx
    max_spikes = -1
    best_start_idx = 0

    for start_idx in range(0, total_steps - T_window_sim_steps + 1):
        start_time = t_vec_exp[start_idx]
        end_time = start_time + T_window
        n_spikes = np.sum((spike_times_post_exp >= start_time) & (spike_times_post_exp <  end_time))

        # if first window with ≥early_return_spikes spikes, return immediately
        if n_spikes >= early_return_spikes:
            end_idx = start_idx + T_window_sim_steps
            t_vec_exp_window = t_vec_exp[start_idx:end_idx]
            V_m_exp_window = V_m_exp[start_idx:end_idx]
            I_inj_exp_window = I_inj_exp[start_idx:end_idx]

            mask = (spike_times_post_exp >= start_time) & (spike_times_post_exp < end_time)
            spike_times_post_exp_window = spike_times_post_exp[mask]
            spike_times_post_exp_window_zeroed = np.asarray(spike_times_post_exp_window) - t_vec_exp_window[0]
            
            return t_vec_exp_window, V_m_exp_window, I_inj_exp_window, spike_times_post_exp_window, spike_times_post_exp_window_zeroed, start_idx, end_idx
    
    
        # otherwise return window with max amount of spikes
        if n_spikes > max_spikes:
            max_spikes = n_spikes
            best_start_idx = start_idx
    
    start_idx = best_start_idx
    end_idx = best_start_idx + T_window_sim_steps
    t_vec_exp_window = t_vec_exp[start_idx:end_idx]
    V_m_exp_window = V_m_exp[start_idx:end_idx]
    I_inj_exp_window = I_inj_exp[start_idx:end_idx]

    start_time = t_vec_exp[start_idx]
    end_time = start_time + T_window
    mask = (spike_times_post_exp >= start_time) & (spike_times_post_exp < end_time)
    spike_times_post_exp_window = spike_times_post_exp[mask]
    spike_times_post_exp_window_zeroed = np.asarray(spike_times_post_exp_window) - t_vec_exp_window[0]

    return t_vec_exp_window, V_m_exp_window, I_inj_exp_window, spike_times_post_exp_window, spike_times_post_exp_window_zeroed, best_start_idx, end_idx
    

def smooth_voltage_trace(V_m, dt_original, sigma=1):
    # smooths the voltage trace using a Gaussian kernel with standard deviation sigma 
    # input
    # V_m is an array of length in ms
    # dt_original is the time resolution of the voltage trace
    # sigma is the std of the Gaussian kernel in ms
    
    # output
    # V_m_smooth is an array of length in ms

    V_m_smooth = gaussian_filter1d(V_m, sigma=sigma/dt_original)
    return V_m_smooth

def downsample_voltage_trace(t_vec, V_m, dt_original=0.1, dt_target_V_m=1.0):
    # downsamples the input trace from dt_original resolution to dt_target resolution using the mean of respective bins
    
    # input
    # t_vec is the time vector in ms
    # V_m is the voltage trace in mV
    # dt_original is the time step of the given voltage trace
    # dt_target_V_m is the target time step
    
    # output
    # t_vec_downsampled is the downsampled time vector in ms
    # V_m_downsampled is the downsampled voltage trace in mV
    
    t_vec_downsampled = t_vec[::int(dt_target_V_m/dt_original)]
    V_m_downsampled = V_m.reshape(-1, int(dt_target_V_m/dt_original)).mean(axis=1)
    
    return t_vec_downsampled, V_m_downsampled

def binarize_spike_times(spike_times_post, t_vec, dt_original=0.1, dt_target_spike_train=100.0):
    # binarizes spike times into a binary spike train at coarser resolution
    
    # input
    # spike_times_post are the spike times in ms
    # t_vec is the time vector in ms
    # dt_original is the resolution of t_vec in ms
    # dt_target_spike_train is the target time step for the spike times in ms

    # output
    # t_vec_binarized is the corresponding time vector in ms
    # spike_train_binarized is an array of length with dt_target_spike_train resolution
    # spike_counts is a list of spike counts in dt_target_spike_train resolution
    # spike_rates is a list of spike rates in dt_target_spike_train resolution in Hz

    total_duration = t_vec[-1] - t_vec[0]
    n_bins = int(np.ceil(total_duration / dt_target_spike_train))
    
    # align spike times to start of t_vec
    spike_times_aligned = spike_times_post - t_vec[0]
    
    # compute bin edges
    bin_edges = np.arange(0, n_bins + 1) * dt_target_spike_train
    
    # histogram
    spike_counts, _ = np.histogram(spike_times_aligned, bins=bin_edges)
    spike_train_binarized = (spike_counts > 0).astype(int)

    # bin centers as time vector
    t_vec_binarized = t_vec[0] + (np.arange(n_bins) + 0.5) * dt_target_spike_train

    spike_rates = np.array(spike_counts) / dt_target_spike_train * 1000 # translate spike count into rates
    
    return t_vec_binarized, spike_train_binarized, spike_counts, spike_rates

def subthreshold_voltage_trace(V_m, V_thresh):
    # cuts all voltages above threshold & sets them to threshold to get subthreshold voltage trace
    
    # input
    # V_m is the voltage trace in mV
    # V_thresh is the voltage threshold in mV

    # output 
    # V_m_subthreshold is the subthreshold voltage trace in mV

    V_m_subthreshold = np.minimum(V_m, V_thresh+0.01) # add a little 
    return V_m_subthreshold

######################## SBI function ########################

def AdExp_I_inj(I_inj, C_m, R_m, E_L, V_thresh, V_reset, Delta_T, tau_w, a, b, tau_ref, membrane_noise=0.0):
    # simulation AdExp model with current injection
    
    # input
    # I_inj is the experimentally injected current in pA
    # C_m is the membrane capacitance in pF
    # R_m is the membrane resistance in MOhm
    # E_L is the resting potential (leak reversal potential) in mV
    # V_thresh is the spike generation threshold in mV
    # V_reset is the reset potential in mV
    # Delta_T is the slope factor in mV
    # tau_w is the adaptation time constant in ms
    # a is the subthreshold adaptation in nS
    # b is the spike-triggered adaptation in nA
    # tau_ref is the absolute refractory period in ms
    # membrane_noise is the standard deviation of the membrane noise in mV/ms

    # output 
    # time is an array of length n of the time in ms
    # V_m is an array of length n of the membrane voltage in mV
    # spike_times_post is a list of postsynaptic spike times in ms

    dt = 0.1*ms
    T = len(I_inj)*dt
    
    start_scope()

    I_inj_timed = TimedArray(I_inj, dt=dt)
    defaultclock.dt = dt

    N = 1  # number of postsynaptic neurons

    # define V_T & spiking threshold
    V_T = V_thresh - 5 * Delta_T
    V_thresh_spike_detection = V_thresh #+ 3 * Delta_T
    
    if membrane_noise == 0.0: 
        eqs = '''
        dV_m/dt = ((E_L - V_m)/R_m + Delta_T/R_m*exp((V_m - V_T)/Delta_T) - w_ad + I_inj_timed(t)) / C_m : volt (unless refractory)
        dw_ad/dt = (a*(V_m - E_L) - w_ad) / tau_w : amp
        '''
    
        neuron = NeuronGroup(N, model=eqs, threshold='V_m > V_thresh_spike_detection', reset='V_m = V_reset; w_ad += b', refractory=tau_ref, method='euler', namespace={'V_thresh_spike_detection': V_thresh_spike_detection})
    
    if membrane_noise > 0.0: 
        eqs = '''
        dV_m/dt = ((E_L - V_m)/R_m + Delta_T/R_m*exp((V_m - V_T)/Delta_T) - w_ad + I_inj_timed(t)) / C_m + membrane_noise*xi*sqrt(dt) : volt (unless refractory)
        dw_ad/dt = (a*(V_m - E_L) - w_ad) / tau_w : amp
        '''
    
        neuron = NeuronGroup(N, model=eqs, threshold='V_m > V_thresh_spike_detection', reset='V_m = V_reset; w_ad += b', refractory=tau_ref, method='euler', namespace={'V_thresh_spike_detection': V_thresh_spike_detection})
    
    # initialize V_m and w_ad
    #I_mean = np.mean(I_inj)  # carries the unit from I_inj pA
    V0 = E_L #+ I_mean * R_m 
    #V0 = np.clip(V0, E_L - 20*mV, V_thresh - 5*mV) 
    w0 = 0 * pA #a * (V0 - E_L)
    
    neuron.V_m = V0 #V_reset
    neuron.w_ad = w0 #0 * pA

    spikes = SpikeMonitor(neuron)
    post_states = StateMonitor(neuron, 'V_m', record=True)
    
    run(T, report='text')

    time = post_states.t / ms
    V_m = post_states.V_m[0] / mV
    spike_times_post = spikes.t / ms

    return time, V_m, spike_times_post

def V_m_statistics(V_m, spike_times):
    # calculate statistics of a voltage trace

    # input
    # V_m is a voltage trace in mV
    # spike_times are spike times in ms

    # output
    # stats_V_m_sum is a list of the summary statistics
    
    V_mean = np.mean(V_m)
    V_median = np.median(V_m)
    V_std = np.std(V_m)
    n_spikes = len(spike_times)

    quantiles=(5, 10, 25, 75, 95) # 50th percentile is median so left out
    percs = np.percentile(V_m, quantiles, method="linear") 

    stats_V_m_sum = [round(V_mean, 2), round(V_median, 2), round(V_std, 2), n_spikes] + percs.tolist()
    stats_V_m_sum = [round(float(x),2) for x in stats_V_m_sum]  # ensure floats
    
    return stats_V_m_sum


def spike_times_statistics(spike_times):
    # calculate statistics of a spike times

    # input
    # spike_times are spike times in ms

    # output
    # stats_spike_times_sum is a list of the summary statistics
    
    n_spikes = float(len(spike_times))

    isi_edges_ms=(1,2,4,8,16,32,64,128,256)

    if n_spikes > 2:
        isi = np.diff(spike_times)
        isi_mean = np.mean(isi)
        isi_std = np.std(isi)
        isi_cv = isi_std / isi_mean
        first_spike = spike_times[0]
        last_spike = spike_times[-1]
        isi_hist, _ = np.histogram(isi, bins=np.array(isi_edges_ms, dtype=np.float32), density=True)
    else:
        isi_mean, isi_std, isi_cv = 0.0, 0.0, 0.0
        first_spike = 0.0
        last_spike = 0.0
        isi_hist = np.zeros(len(isi_edges_ms)-1, dtype=np.float32)

    stats_spike_times_sum = [n_spikes, round(isi_mean, 2), round(isi_std, 2), round(isi_cv,2), round(first_spike, 2), round(last_spike, 2)] + isi_hist.tolist()
    return stats_spike_times_sum

def calculate_STA(V_m, spike_times_post, V_thresh, pre_spike_time_window=5.0, dt_original=0.1, std_ms=0.5):
    # return spike triggering voltage traces & spike triggered average (STA) with spikes occuring within pre_spike_time_window after a previous spike being dropped
    # input
    # V_m is the simulted voltage trace in mV
    # spike_times_post are the spike times in ms
    # V_thresh is the threshold value to be used as spike cut off in mV
    # pre_spike_time_window is the time window to calculate the STA in ms
    # dt_original is the time step of the given voltage trace in ms
    # std_ms is the standard deviation of smoothing of the STAs by a Gaussian kernel in ms
        
    # output
    # STA is the spike triggered average in mV
    # STA_segs is a list of spike triggering voltage excerpts in mV
    
    pre_spike_time_steps_window = int(np.round(pre_spike_time_window / dt_original)) # get number of time steps of pre_spike_time_steps_window
    spike_times_post_array = np.asarray(spike_times_post, dtype=float)

    # in case voltage trace does not contain any spikes, return empty results
    if spike_times_post_array.size == 0:
        pre_spike_time_steps_window = int(pre_spike_time_window/dt_original)
        STA_segs = np.zeros(pre_spike_time_steps_window, dtype=np.float32)
        STA = np.zeros(pre_spike_time_steps_window, dtype=np.float32)
        return STA, STA_segs
    
    # only use spike_times_post elements with an ISI larger than 10 ms to previous spike 
    spike_times_post_array_valid = np.ones_like(spike_times_post_array, dtype=bool) 
    if spike_times_post_array.size > 1: 
        spike_times_post_array_valid[1:] = np.diff(spike_times_post_array) >= float(pre_spike_time_window + 10) 
    
    spike_times_post_array_used = spike_times_post_array[spike_times_post_array_valid]
    
    STA_segs_list = []
    
    # only use spike_times_post elements with an ISI larger than 10 ms to previous spike
    spike_times_post_array_valid = np.ones_like(spike_times_post_array, dtype=bool)
    if spike_times_post_array.size > 1:
        spike_times_post_array_valid[1:] = np.diff(spike_times_post_array) >= float(pre_spike_time_window + 10.0)
    spike_times_post_array_used = spike_times_post_array[spike_times_post_array_valid]
    
    STA_segs_list = []
    V_thresh_add = float(V_thresh) - 0.1

    for spike in spike_times_post_array_used:
        spike_idx = int(np.round(spike / dt_original))
        higher_idx = spike_idx
        lower_idx  = higher_idx - pre_spike_time_steps_window

        # require a complete pre-window strictly inside the signal
        if lower_idx < 0 or higher_idx > len(V_m):
            continue

        # work on one segment so the two arrays align
        seg = V_m[lower_idx:higher_idx]
        if seg.size < 2:
            continue

        # rising crossing within the window: seg[:-1] < V_thresh_add and seg[1:] >= V_thresh_add
        crosses = np.where((seg[:-1] < V_thresh_add) & (seg[1:] >= V_thresh_add))[0]

        if crosses.size > 0:
            # last rising crossing; +2 because crosses is in seg[:-1]
            higher_idx = lower_idx + int(crosses[-1]) + 2
            lower_idx  = higher_idx - pre_spike_time_steps_window
            # re-check bounds after adjusting
            if lower_idx < 0 or higher_idx > len(V_m):
                continue
            seg = V_m[lower_idx:higher_idx]  # refresh exact window

        # if still wrong length, skip
        if seg.size != pre_spike_time_steps_window:
            continue

        STA_segs_list.append(seg.astype(np.float32))
    
    # OLD part, with fewer safties
    """
    for spike in spike_times_post_array_used:

        spike_idx = int(np.round(spike / dt_original))
        lower_idx = max(0, spike_idx - pre_spike_time_steps_window)
        higher_idx = spike_idx
        
        cross = np.where((V_m[lower_idx:higher_idx-1] < V_thresh_add) & (V_m[lower_idx+1:higher_idx] >= V_thresh_add))[-1]
        higher_idx = (lower_idx + cross[-1] +2) if cross.size else higher_idx
        lower_idx  = higher_idx - pre_spike_time_steps_window

        #print(higher_idx) # checked that voltage trace has quite large jumps (up to 10 mV)
        
        if lower_idx >= 0 and higher_idx <= len(V_m):
            V_m_STA_raw = V_m[lower_idx:higher_idx]
            #V_m_STA_smoothed = gaussian_filter1d(V_m[lower_idx:higher_idx], sigma=std_ms / dt_original)
            #STA_segs_list.append(V_m_STA_raw)
            #STA_segs_list.append(V_m_STA_smoothed)
            """
    
    # in case no spike triggering segments can be extracted, return empty results
    if len(STA_segs_list) == 0:
        pre_spike_time_steps_window = int(pre_spike_time_window/dt_original)
        STA_segs = np.zeros(pre_spike_time_steps_window, dtype=np.float32)
        STA = np.zeros(pre_spike_time_steps_window, dtype=np.float32)
        return STA, STA_segs

    STA_segs = np.stack(STA_segs_list, axis=0)
    STA = np.mean(STA_segs, axis=0)
    STA_smoothed = gaussian_filter1d(STA, sigma=std_ms / dt_original, mode="nearest")
           
    return STA_smoothed.astype(np.float32), STA_segs

def calculate_ASA(V_m, spike_times_post, V_thresh, post_spike_time_window=20.0, dt_original=0.1, std_ms=0.5, tau_ref=0):
    # return after-spiking voltage traces & after-spike average (ASA); spikes with too-small ISI to the next spike are dropped
    
    # input
    # V_m is the simulated or experimental voltage trace in mV
    # spike_times_post are the spike times in ms
    # V_thresh is the threshold value in mV
    # post_spike_time_window is the time window after the spike used for the ASA in ms
    # dt_original is the time step of the given voltage trace in ms
    # std_ms is the standard deviation of Gaussian smoothing applied to the averaged ASA in ms
    # tau_ref is the absolute refractory period after a spike in ms to achor the ASA beginning
    
    # output
    # ASA is the after-spike average in mV
    # ASA_segs is a list of after-spike voltage excerpts in mV

    post_spike_time_steps_window = int(np.round(post_spike_time_window / dt_original))  # number of time steps in the post window
    spike_times_post_array = np.asarray(spike_times_post)

    # in case voltage trace does not contain any spikes, return empty results
    if spike_times_post_array.size == 0:
        post_spike_time_steps_window = int(post_spike_time_window / dt_original)
        ASA_segs = np.zeros(post_spike_time_steps_window)
        ASA = np.zeros(post_spike_time_steps_window)
        return ASA, ASA_segs
    
    # only use spike_times_post elements whose NEXT ISI is large enough to hold the post window (+10 ms safety)
    spike_times_post_array_valid = np.ones_like(spike_times_post_array, dtype=bool)
    if spike_times_post_array.size > 1:
        next_isi = np.diff(spike_times_post_array)
        spike_times_post_array_valid[:-1] = next_isi >= float(post_spike_time_window + tau_ref + 10.0)
    spike_times_post_array_used = spike_times_post_array[spike_times_post_array_valid]

    # the anchor is the first sample after the spike with V_m <= V_thresh
    ASA_segs_list = []

    for spike in spike_times_post_array_used:
        spike_idx = int(np.round(spike / dt_original)) + int(np.ceil((float(spike) + float(tau_ref)) / dt_original)) # add absolute refractory period translated into index

        # search forward from the spike to find the first falling crossing: V[k] > V_thresh and V[k+1] <= V_thresh
        # limit the search so that the full post window can still fit
        search_lo = spike_idx
        search_hi = min(len(V_m) - 2, spike_idx + post_spike_time_steps_window)  # -2 so k+1 is valid

        if search_hi < search_lo:
            continue

        cond = (V_m[search_lo:search_hi+1] > V_thresh) & (V_m[search_lo+1:search_hi+2] <= V_thresh)
        cross_rel = np.flatnonzero(cond)

        if cross_rel.size > 0:
            reset_idx = search_lo + int(cross_rel[0]) + 1   # first sample <= threshold AFTER the spike
        else:
            # fallback: first index j >= spike_idx with V_m[j] <= V_thresh
            j = spike_idx
            while j < len(V_m) and V_m[j] > V_thresh:
                j += 1
            reset_idx = j

        start_idx = reset_idx
        end_idx   = start_idx + post_spike_time_steps_window  # exclusive

        # keep only complete segments within bounds
        if start_idx < 0 or end_idx > len(V_m):
            continue

        V_m_ASA_raw = V_m[start_idx:end_idx]
        ASA_segs_list.append(V_m_ASA_raw)

    # in case no after-spike segments can be extracted, return empty results
    if len(ASA_segs_list) == 0:
        post_spike_time_steps_window = int(post_spike_time_window / dt_original)
        ASA_segs = np.zeros(post_spike_time_steps_window)
        ASA = np.zeros(post_spike_time_steps_window)
        return ASA, ASA_segs

    ASA_segs = np.stack(ASA_segs_list, axis=0)
    ASA = np.mean(ASA_segs, axis=0)
    ASA_smoothed = gaussian_filter1d(ASA, sigma=std_ms / dt_original, mode="nearest")
    return ASA_smoothed, ASA_segs

def process_voltage_trace_spike_times_post(t_vec, V_m, spike_times_post, V_thresh, dt_original, dt_target_V_m, dt_target_spike_train):
    # smooth & downsample a voltage trace & binarize spike times
    
    # input
    # t_vec is the time vector in ms
    # V_m is the simulted voltage trace in mV
    # spike_times_post are the spike times in ms
    # V_thresh is the threshold in mV used as spike cut off
    # dt_original is the time step of the given voltage trace in ms
    # dt_target_V_m is the target time step for the voltage trace in ms
    # dt_target_spike_train is the target time step for the spike times in ms

    # output
    # t_vec_downsampled is the downsampled time vector in ms
    # t_vec_binarized is the time vector of the binarization of the spike train in ms
    # V_m_smooth_downsampled is the smoothed & downsampled voltage trace in dt_target_V_m resolution mV
    # spike_train_binarized is the binarized spike train in dt_target_spike_train resolution
    # spike_train_counts is a list of spike counts in dt_target_spike_train resolution
    # spike_train_rates is a list of spike rates in dt_target_spike_train resolution in Hz

    V_m_subthreshold = subthreshold_voltage_trace(V_m, V_thresh)
    V_m_subthreshold_smooth = smooth_voltage_trace(V_m_subthreshold, dt_original=dt_original, sigma=1)
    t_vec_downsampled, V_m_subthreshold_smooth_downsampled = downsample_voltage_trace(t_vec, V_m_subthreshold_smooth, dt_original=dt_original, dt_target_V_m=dt_target_V_m)
    t_vec_binarized, spike_train_binarized, spike_train_counts, spike_train_rates = binarize_spike_times(spike_times_post, t_vec, dt_original=dt_original, dt_target_spike_train=dt_target_spike_train)

    return t_vec_downsampled, t_vec_binarized, V_m_subthreshold_smooth_downsampled, spike_train_binarized, spike_train_counts, spike_train_rates

def define_reasonable_V_reset(p_V_reset, E_L, V_thresh):
    # translate the 
    # input
    # p_V_reset is the ratio of V_reset to lie between E_L-10 and V_thresh-1
    # E_L is the resting potential in mV
    # V_thresh is the threshold potential in mV

    # output 
    # V_reset is the reset potential in mV

    if p_V_reset == float:
        p_V_reset = np.clip(float(p_V_reset), 0.0, 1.0)
    V_reset = (1-p_V_reset) * (E_L-10) + p_V_reset * (V_thresh-1)
    return V_reset

def translate_p_to_V_reset(p_V_reset_map, p_V_reset_mean, p_V_reset_std, E_L, V_thresh):
    # translate map, mean & std of p_V_reset to map, mean & std of V_reset
    # input
    # p_V_reset_map, p_V_reset_mean, p_V_reset_std are the map, mean & std of p_V_reset estimated by SBI
    # E_L is the leak potential
    # V_thresh is the threshold potential

    # output
    # V_reset_map, V_reset_mean, V_reset_std are the map, mean & std of V_reset in mV
    
    # estimate map & mean
    V_reset_map  = define_reasonable_V_reset(p_V_reset_map,  E_L, V_thresh)
    V_reset_mean = define_reasonable_V_reset(p_V_reset_mean, E_L, V_thresh)
    # approximate std mapping
    V_reset_std  = p_V_reset_std * abs(V_reset_mean - define_reasonable_V_reset(min(max(p_V_reset_mean+1e-3,0.0),1.0), E_L, V_thresh)) / 1e-3
    return V_reset_map, V_reset_mean, V_reset_std

######################## sequential SBI function ########################


def build_s1_features(V_m, spike_times_post, V_thresh_est, dt_original=0.1, pre_spike_time_window=5.0, std_ms=0.5):
    # build features for stage 1 with V_m statistics (V_mean, V_median, V_std, n_spikes) + STA
    
    # input
    # V_m is the simulated or experimental voltage trace in mV
    # spike_times_post are the spike times in ms
    # V_thresh_est is the estimated threshold in mV used as spike cut off for STA alignment
    # dt_original is the time step of the given voltage trace in ms
    # pre_spike_time_window is the time window for STA in ms
    # std_ms is the standard deviation for Gaussian smoothing of the STAs in ms
    
    # output
    # feat1 is the feature array
    V_m_stats = V_m_statistics(V_m, spike_times_post)
    V_m_stats_tensor = np.asarray(V_m_stats, dtype=np.float32)
    STA, STA_segs = calculate_STA(V_m, spike_times_post, V_thresh_est, dt_original=dt_original, pre_spike_time_window=pre_spike_time_window, std_ms=std_ms)
    STA_tensor = STA.astype(np.float32)
    # smooth and downsample V_m --> makes SNPE unstable
    #t_vec = np.arange(len(V_m)) * dt_original
    #t_vec_down, _, V_m_subthreshold_smooth_downsampled, _, _, _ = process_voltage_trace_spike_times_post(t_vec, V_m, spike_times_post, V_thresh=V_thresh_est, dt_original=dt_original, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train)

    # translate into tensor 
    #V_m_tensor = torch.tensor(V_m_subthreshold_smooth_downsampled, dtype=torch.float32)
    #t_vec, dt_target_V_m, dt_target_spike_train
    #_, t_vec_binarized, _, spike_train_binarized, spike_train_counts, spike_train_rates = process_voltage_trace_spike_times_post(t_vec, V_m, spike_times_post, V_thresh_est, dt_original=dt_original, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train)
    

    #feat1 = np.concatenate([V_m_stats_tensor, STA_tensor, V_m_tensor], axis=0)
    feat1 = np.concatenate([V_m_stats_tensor, STA_tensor], axis=0)
    return feat1

def build_s2_features(V_m, spike_times_post, V_thresh_est, t_vec, post_spike_time_window=20, dt_original=0.1, std_ms=0.5, dt_target_V_m=2.0, dt_target_spike_train=125.0, tau_ref=0):
    # build features for stage 2 with last six entries of stats_sum (n_spikes, isi_mean, isi_std, isi_cv, first_spike, last_spike)
    # input
    # V_m is the simulated or experimental voltage trace in mV
    # spike_times_post are the spike times in ms
    # V_thresh_est is the estimated threshold in mV used as spike cut off for ASA alignment
    # t_vec is the time vector in ms
    # post_spike_time_window is the time window for ASA in ms (default: 5.0)
    # dt_original is the time step of the given voltage trace in ms
    # std_ms is the standard deviation (ms) for Gaussian smoothing of the STAs
    # dt_target_V_m is the target time step for the voltage trace in ms
    # dt_target_spike_train is the target time step for the spike times in ms
    
    # output
    # feat2 is the feature array
    
    stats = np.asarray(spike_times_statistics(spike_times_post), dtype=np.float32)
    ASA, _ = calculate_ASA(V_m, spike_times_post, V_thresh=V_thresh_est, post_spike_time_window=post_spike_time_window, dt_original=dt_original, std_ms=std_ms, tau_ref=tau_ref)
    _, t_vec_binarized, _, spike_train_binarized, spike_train_counts, spike_train_rates = process_voltage_trace_spike_times_post(t_vec, V_m, spike_times_post, V_thresh_est, dt_original=dt_original, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train)
    feat2 = np.concatenate([stats, ASA.astype(np.float32), spike_train_rates.astype(np.float32)], axis=0)
    return feat2


# simulators for stage 1 and stage 2

def simulator_wrapper_s1(params_tensor, I_inj_exp, E_L_est, V_thresh_est, tau_ref, dt_original=0.1, pre_spike_time_window=5.0, std_ms=0.5):
    # simulator wrapper for SBI inference of stage 1

    # input
    # params_tensor is a torch tensor [R_m, C_m, Delta_T]
    # I_inj_exp is the experimentally injected current in pA
    # E_L_est is the estimated leak/resting potential in mV
    # V_thresh_est is the estimated threshold in mV
    # tau_ref is the absolute refractory period in ms
    # dt_original is the time step of the given voltage trace in ms
    # pre_spike_time_window is the STA window in ms
    # std_ms is the STA smoothing std in ms
    
    # output
    # feat is the feature tensor
    
    # params_tensor: [R_m, C_m, V_thresh_est, Delta_T]
    # other AdExp parameters fixed to weak adaptation to focus identifiability
    R_m_b     = params_tensor[0].item()*Mohm
    C_m_b     = params_tensor[1].item()*pF
    #V_thresh_b= params_tensor[2].item()*mV
    #Delta_T_b = params_tensor[3].item()*mV
    V_thresh_b = V_thresh_est*mV
    Delta_T_b = params_tensor[2].item()*mV
    
    # fixed params for stage 1 (weak adaptation; not inferred here)
    E_L_b   = E_L_est*mV
    V_reset_b = E_L_est*mV
    tau_w_b = 200*ms
    a_b     = 2*nS
    b_b     = 0.02*nA

    tau_ref_b = tau_ref*ms

    I_inj_exp_b = I_inj_exp*pA
    time_sim, V_m_sim, spike_times_post_sim = AdExp_I_inj(I_inj_exp_b, C_m_b, R_m_b, E_L_b, V_thresh_b, V_reset_b, Delta_T_b, tau_w_b, a_b, b_b, tau_ref_b)
    
    # Stage-1 feature: [V_mean, V_median, V_std] + STA
    # use current V_thresh candidate to align STA crossing; fall back to provided experimental threshold if given
    #V_thresh_for_STA_local = params_tensor[2].item() if V_thresh_for_STA is None else V_thresh_for_STA
    
    feat_np = build_s1_features(V_m_sim, spike_times_post_sim, V_thresh_est, dt_original=dt_original, pre_spike_time_window=pre_spike_time_window, std_ms=std_ms)
    feat = torch.tensor(feat_np, dtype=torch.float32)
    return feat

def simulator_wrapper_s2(params_tensor, I_inj_exp, E_L_est, R_m_fix, C_m_fix, V_thresh_est, Delta_T_fix, tau_ref, t_vec, post_spike_time_window, dt_original, std_ms, dt_target_V_m, dt_target_spike_train):
    # simulator wrapper for SBI inference of stage 2
    
    # input
    # params_tensor is the torch tensor [p_V_reset, tau_w, a_w, b_w] (dimensionless, ms, nS, nA)
    # I_inj_exp is the experimentally injected current in pA
    # E_L_est is the estimated leak/resting potential in mV
    # V_thresh_est is the estimated threshold potential in mV
    # R_m_fix, C_m_fix , Delta_T_fix are the s1 fixed MAPs (MOhm, pF, mV)
    # tau_ref is the absolute refractory period in ms
    # dt_original is the time step of the given voltage trace in ms
    # dt_target is the target time step in ms
    # t_vec is the time vector in ms
    # post_spike_time_window is the time window for ASA in ms (default: 5.0)
    # dt_original is the time step of the given voltage trace in ms
    # std_ms is the standard deviation (ms) for Gaussian smoothing of the STAs
    # dt_target_V_m is the target time step for the voltage trace in ms
    # dt_target_spike_train is the target time step for the spike times in ms
    
    # output
    # feat is the feature tensor
    
    # params_tensor  [p_V_reset, tau_w, a_w, b_w]
    
    p_V_reset_b = float(params_tensor[0].item())
    tau_w_b     = params_tensor[1].item()*ms
    a_b         = params_tensor[2].item()*nS
    b_b         = params_tensor[3].item()*nA

    # fixed from Stage 1 MAPs
    E_L_b     = E_L_est*mV
    R_m_b     = R_m_fix*Mohm
    C_m_b     = C_m_fix*pF
    V_thresh_b= V_thresh_est*mV
    Delta_T_b = Delta_T_fix*mV

    # map p_V_reset -> V_reset 
    V_reset_val = define_reasonable_V_reset(p_V_reset_b, E_L_est, V_thresh_est)
    V_reset_b = V_reset_val*mV

    tau_ref_b = tau_ref*ms

    I_inj_exp_b = I_inj_exp*pA
    time_sim, V_m_sim, spike_times_post_sim = AdExp_I_inj(I_inj_exp_b, C_m_b, R_m_b, E_L_b, V_thresh_b, V_reset_b, Delta_T_b, tau_w_b, a_b, b_b, tau_ref_b)
    # Stage-2 feature: spike times stats + ASA + binarized spike train
    feat_np = build_s2_features(V_m_sim, spike_times_post_sim, V_thresh_est, t_vec, post_spike_time_window=post_spike_time_window, dt_original=dt_original, std_ms=std_ms, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train, tau_ref=tau_ref)
    feat = torch.tensor(feat_np, dtype=torch.float32)
    return feat


# main sequential SBI
def run_sbi_sequential(V_m_exp, I_inj_exp, spike_times_post_exp, E_L_est, V_thresh_est, tau_ref, t_vec, dt_original, dt_target_V_m, dt_target_spike_train, pre_spike_time_window, post_spike_time_window, std_ms, T_window, num_simulations_s1=20000, num_simulations_s2=20000, savename_s1='seq_s1', savename_s2='seq_s2', use_cached_s1=False, use_cached_s2=False, *args, **kwargs):
    # performs sequential SBI to estimate AdExp parameters in two stages
    # stage 1 (s1) uses voltage statistics (first three entries of stats_sum: V_mean, V_median, V_std) and the spike-triggered average (STA) to estimate R_m, C_m, V_thresh, Delta_T
    # stage 2 (s2) fixes s1 MAPs and uses ISI statistics (last six entries of stats_sum: n_spikes, isi_mean, isi_std, isi_cv, first_spike, last_spike) together with the binarized spike train to estimate V_reset, tau_w, a_w, b_w (V_reset inferred via p_V_reset mapping)
    
    # input
    # V_m_exp is the experimental membrane voltage in mV
    # I_inj_exp is the experimentally injected current in pA
    # spike_times_post_exp is the experimentally measured spike times in ms
    # E_L_est is the estimated resting potential in mV
    # V_thresh_est is the estimated spiking threshold in mV (for STA alignment on experimental data)
    # tau_ref is the absolute refractory period in ms
    # t_vec is the time vector in ms
    # dt_original is the resolution of the voltage trace in ms
    # dt_target_V_m is the target time step for the voltage trace in ms
    # dt_target_spike_train is the target time step for the spike times in ms
    # pre_spike_time_window is the STA window in ms
    # post_spike_time_window is the time window for ASA in ms (default: 5.0)
    # std_ms is the standard deviation (ms) for Gaussian smoothing of the STAs & ASAs
    # T_window is the simulation time in ms (kept for symmetry; not directly used here)
    # num_simulations_s1 is the number of simulations for s1 posterior (default: 20000)
    # num_simulations_s2 is the number of simulations for s2 posterior (default: 20000)
    # savename_s1/savename_s2 is the cache filenames (without path) for saving/loading simulated data
    # use_cached is the if True, loads cached simulations from SBI_logs
    
    # output
    # maps are the maximum a posteriori estimated values 
    # means are the mean estimated values 
    # stds are the standard deviations of the estimated values
    # samples are samples of the posterior distribution
    
    # timeout helpers (sampling only)
    class _SBITimeout(Exception): ...
    def _timeout_handler(signum, frame): raise _SBITimeout
    # install once per worker process (safe to call again)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
    except Exception:
        # On some platforms signals may not be available; just don't time out.
        pass

    _TIME_LIMIT_SEC = int(os.getenv("SBI_TIME_LIMIT_SEC", "1200"))  # 20 minutes

    def _with_timeout(seconds, fn, *f_args, **f_kwargs):
        """Run fn(*args, **kwargs) with SIGALRM timeout; raises _SBITimeout on expiry."""
        if hasattr(signal, "alarm"):
            prev = signal.alarm(0)
            try:
                signal.alarm(int(seconds))
                return fn(*f_args, **f_kwargs)
            finally:
                signal.alarm(0)
                if prev:
                    signal.alarm(prev)
        else:
            # Fallback: no timeout support; just run
            return fn(*f_args, **f_kwargs)
        
    
    # prepare experimental feature vectors
    # stage 1 features from subthreshold stats + STA (use experimental threshold estimate for STA)
    feat_exp_s1_np = build_s1_features(V_m_exp, spike_times_post_exp, V_thresh_est, dt_original=dt_original, pre_spike_time_window=pre_spike_time_window, std_ms=std_ms)
    x_exp_s1 = torch.tensor(feat_exp_s1_np, dtype=torch.float32)
    # stage 2 features from ISI stats + ASA + binarized spikes    
    feat_exp_s2_np = build_s2_features(V_m_exp, spike_times_post_exp, V_thresh_est, t_vec, post_spike_time_window=post_spike_time_window, dt_original=dt_original, std_ms=std_ms, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train, tau_ref=tau_ref)
    x_exp_s2 = torch.tensor(feat_exp_s2_np, dtype=torch.float32)

    # stage 1: infer R_m, C_m, Delta_T 
    # priors focused on passive/threshold params; others fixed in simulator

    # define prior
    ## not included 0) E_L: (E_L_est +- 5) mV
    # 1) R_m: 25-500 MOhm # from DynIV
    # 2) C_m: 2-120 pF # from DynIV but extended after playing around
    ## not inlcuded 3) V_thresh_est: (E_L+5) - (E_L+25) mV # alternatively V_thresh_est
    # 4) Delta_T: 0.5 - 5 mV
    
    prior_s1 = sbi_utils.BoxUniform(
        #low = torch.tensor([25.0,  2.0, (V_thresh_est-8.0),  0.5], dtype=torch.float32),
        #high= torch.tensor([500.0, 120.0, (V_thresh_est+1.0), 5.0], dtype=torch.float32),)
        low = torch.tensor([25.0,  2.0,  0.5], dtype=torch.float32),
        high= torch.tensor([500.0, 120.0, 5.0], dtype=torch.float32),)
    
    simulator_s1 = partial(simulator_wrapper_s1, I_inj_exp=I_inj_exp, E_L_est=E_L_est, V_thresh_est=V_thresh_est, tau_ref=tau_ref, dt_original=dt_original, pre_spike_time_window=pre_spike_time_window, std_ms=std_ms) # let the candidate param define STA crossing

    if not use_cached_s1:
        theta_s1 = prior_s1.sample((num_simulations_s1,))
        x_s1 = torch.stack([simulator_s1(param) for param in theta_s1])
        inference_s1 = sbi_inference.SNPE(prior=prior_s1)
        density_estimator_s1 = inference_s1.append_simulations(theta_s1, x_s1).train()
        #posterior_s1 = inference_s1.build_posterior(density_estimator_s1)
        posterior_s1_rej = inference_s1.build_posterior(density_estimator_s1, sample_with="direct")
        posterior_s1_mcmc = inference_s1.build_posterior(density_estimator_s1, sample_with="mcmc", mcmc_method="slice_np", mcmc_parameters={"num_chains": 1, "thin": 10, "warmup_steps": 500})

        if savename_s1 is not None:
            os.makedirs("../Data_SBI_logs", exist_ok=True)
            with open(f"../Data_SBI_logs/{savename_s1}_n{num_simulations_s1}.pkl", "wb") as f:
                pkl.dump({"theta": theta_s1, "x": x_s1}, f)
    else:
        with open(f"../Data_SBI_logs/{savename_s1}_n{num_simulations_s1}.pkl", "rb") as f:
            s1 = pkl.load(f)
        theta_s1 = s1["theta"]; x_s1 = s1["x"]
        inference_s1 = sbi_inference.SNPE(prior=prior_s1)
        density_estimator_s1 = inference_s1.append_simulations(theta_s1, x_s1).train()
        #posterior_s1 = inference_s1.build_posterior(density_estimator_s1)
        posterior_s1_rej = inference_s1.build_posterior(density_estimator_s1, sample_with="direct")
        posterior_s1_mcmc = inference_s1.build_posterior(density_estimator_s1, sample_with="mcmc", mcmc_method="slice_np", mcmc_parameters={"num_chains": 1, "thin": 10, "warmup_steps": 500})

    
    # stage 1 sampling without timeout + fallback
    #samples_s1 = posterior_s1.sample((1000,), x=x_exp_s1)
    #logp_s1 = posterior_s1.log_prob(samples_s1, x=x_exp_s1)
    #best_idx_s1 = torch.argmax(logp_s1)
    #maps_s1 = samples_s1[best_idx_s1].tolist()
    #means_s1 = samples_s1.mean(dim=0).numpy()
    #stds_s1  = samples_s1.std(dim=0).numpy()
    
    # stage 1 sampling with timeout + fallback
    samples_s1 = None
    used_method_s1 = "direct"
    try:
        samples_s1 = _with_timeout(
            _TIME_LIMIT_SEC,
            posterior_s1_rej.sample,  # only sampling is timed
            (1000,), x=x_exp_s1, show_progress_bars=False)
        print("[SBI] Stage 1: direct sampling finished within time limit.")
    except _SBITimeout:
        print("[SBI][TIMEOUT] Stage 1: direct too slow → switching to MCMC.")
        used_method_s1 = "mcmc"
        try:
            samples_s1 = _with_timeout(
                _TIME_LIMIT_SEC,
                posterior_s1_mcmc.sample,
                (1000,), x=x_exp_s1, show_progress_bars=False)
            print("[SBI] Stage 1: MCMC finished within time limit.")
        except _SBITimeout:
            print("[SBI][TIMEOUT] Stage 1: MCMC also timed out → using NaNs for stage 1.")
            samples_s1 = None
            used_method_s1 = "timeout"

    if samples_s1 is None:
        means_s1 = np.array([np.nan, np.nan, np.nan])
        stds_s1  = np.array([np.nan, np.nan, np.nan])
        maps_s1  = [np.nan, np.nan, np.nan]
        
        # Stage 2 cannot proceed without stage 1 MAPs:
        means_s2 = np.array([np.nan, np.nan, np.nan, np.nan])
        stds_s2  = np.array([np.nan, np.nan, np.nan, np.nan])
        maps_s2  = [np.nan, np.nan, np.nan, np.nan]
        samples = None
        means = np.concatenate([means_s1, means_s2])
        stds  = np.concatenate([stds_s1,  stds_s2])
        maps  = maps_s1 + maps_s2
        return means, stds, maps, samples

    # compute s1 stats
    logp_s1 = posterior_s1_rej.log_prob(samples_s1, x=x_exp_s1) if used_method_s1 == "direct" else posterior_s1_mcmc.log_prob(samples_s1, x=x_exp_s1)
    best_idx_s1 = torch.argmax(logp_s1)
    maps_s1 = samples_s1[best_idx_s1].tolist()
    means_s1 = samples_s1.mean(dim=0).cpu().numpy()
    stds_s1  = samples_s1.std(dim=0).cpu().numpy()


    #R_m_map, C_m_map, Delta_T_map = maps_s1
    R_m_map, C_m_map, Delta_T_map = maps_s1
    print(f"Stage 1 MAP: R_m = {R_m_map:.2f} MOhm, C_m = {C_m_map:.2f} pF, Delta_T = {Delta_T_map:.2f} mV")

    # stage 2: infer V_reset, tau_w, a_w, b_w
    # infer via (p_V_reset, tau_w, a, b) and report V_reset

    # 5) V_reset: (E_L-10) - (V_thresh_est-1) mV --> define between 0 - 1 & make 0 (E_L-10) & make 1 (V_thresh_est-1) & interpolation between both. This prevents the reset to be higher than threshold
    # 6) tau_w: 5-500 ms
    # 7) a: 1-10 nS
    # 8) b: 0.001 - 0.1 nA

    prior_s2 = sbi_utils.BoxUniform(
        low = torch.tensor([0.0,   5.0, 1.0,   0.001], dtype=torch.float32),  # p_V_reset, tau_w, a, b
        high= torch.tensor([1.0, 500.0, 10.0,  0.1  ], dtype=torch.float32),)
    
    simulator_s2 = partial(simulator_wrapper_s2, I_inj_exp=I_inj_exp, E_L_est=E_L_est, R_m_fix=R_m_map, C_m_fix=C_m_map, V_thresh_est=V_thresh_est, Delta_T_fix=Delta_T_map, tau_ref=tau_ref, t_vec=t_vec, post_spike_time_window=post_spike_time_window, dt_original=dt_original, std_ms=std_ms, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train)

    if not use_cached_s2:
        theta_s2 = prior_s2.sample((num_simulations_s2,))
        x_s2 = torch.stack([simulator_s2(param) for param in theta_s2])
        inference_s2 = sbi_inference.SNPE(prior=prior_s2)
        density_estimator_s2 = inference_s2.append_simulations(theta_s2, x_s2).train()
        #posterior_s2 = inference_s2.build_posterior(density_estimator_s2)
        posterior_s2_rej = inference_s2.build_posterior(density_estimator_s2, sample_with="direct")
        posterior_s2_mcmc = inference_s2.build_posterior(density_estimator_s2, sample_with="mcmc", mcmc_method="slice_np", mcmc_parameters={"num_chains": 1, "thin": 10, "warmup_steps": 500})

        if savename_s2 is not None:
            os.makedirs("../Data_SBI_logs", exist_ok=True)
            with open(f"../Data_SBI_logs/{savename_s2}_n{num_simulations_s2}.pkl", "wb") as f:
                pkl.dump({"theta": theta_s2, "x": x_s2}, f)
    else:
        with open(f"../Data_SBI_logs/{savename_s2}_n{num_simulations_s2}.pkl", "rb") as f:
            s2 = pkl.load(f)
        theta_s2 = s2["theta"]; x_s2 = s2["x"]
        inference_s2 = sbi_inference.SNPE(prior=prior_s2)
        density_estimator_s2 = inference_s2.append_simulations(theta_s2, x_s2).train()
        #posterior_s2 = inference_s2.build_posterior(density_estimator_s2)
        posterior_s2_rej = inference_s2.build_posterior(density_estimator_s2, sample_with="direct")
        posterior_s2_mcmc = inference_s2.build_posterior(density_estimator_s2, sample_with="mcmc", mcmc_method="slice_np", mcmc_parameters={"num_chains": 1, "thin": 10, "warmup_steps": 500})
        
    # stage 1 sampling without timeout + fallback
    #samples_s2 = posterior_s2.sample((1000,), x=x_exp_s2)
    #logp_s2 = posterior_s2.log_prob(samples_s2, x=x_exp_s2)
    #best_idx_s2 = torch.argmax(logp_s2)
    #maps_s2 = samples_s2[best_idx_s2].tolist()
    #means_s2 = samples_s2.mean(dim=0).numpy()
    #stds_s2  = samples_s2.std(dim=0).numpy()
    
    # stage 1 sampling with timeout + fallback
    samples_s2 = None
    used_method_s2 = "direct"
    try:
        samples_s2 = _with_timeout(
            _TIME_LIMIT_SEC,
            posterior_s2_rej.sample,
            (1000,), x=x_exp_s2, show_progress_bars=False
        )
        print("[SBI] Stage 2: direct sampling finished within time limit.")
    except _SBITimeout:
        print("[SBI][TIMEOUT] Stage 2: direct too slow → switching to MCMC.")
        used_method_s2 = "mcmc"
        try:
            samples_s2 = _with_timeout(
                _TIME_LIMIT_SEC,
                posterior_s2_mcmc.sample,
                (1000,), x=x_exp_s2, show_progress_bars=False
            )
            print("[SBI] Stage 2: MCMC finished within time limit.")
        except _SBITimeout:
            print("[SBI][TIMEOUT] Stage 2: MCMC also timed out → using NaNs for stage 2.")
            samples_s2 = None
            used_method_s2 = "timeout"

    if samples_s2 is None:
        means_s2 = np.array([np.nan, np.nan, np.nan, np.nan])
        stds_s2  = np.array([np.nan, np.nan, np.nan, np.nan])
        maps_s2  = [np.nan, np.nan, np.nan, np.nan]
        samples = None  # mixed sampling sizes/types → keep memory low and signal "no samples"
        # pack and return:
        means = np.concatenate([means_s1, means_s2])
        stds  = np.concatenate([stds_s1,  stds_s2])
        maps  = maps_s1 + maps_s2
        return means, stds, maps, samples

    # compute s2 stats (and map p_V_reset → V_reset mV)
    logp_s2 = posterior_s2_rej.log_prob(samples_s2, x=x_exp_s2) if used_method_s2 == "direct" else posterior_s2_mcmc.log_prob(samples_s2, x=x_exp_s2)
    best_idx_s2 = torch.argmax(logp_s2)
    maps_s2 = samples_s2[best_idx_s2].tolist()
    means_s2 = samples_s2.mean(dim=0).cpu().numpy()
    stds_s2  = samples_s2.std(dim=0).cpu().numpy()

    # map p_V_reset statistics into V_reset stats for reporting
    def map_vreset_stats(mean_p, std_p, E_L_est_local, V_thresh_local):
        # approximate mean & std mapping
        V_reset_mean = define_reasonable_V_reset(mean_p, E_L_est_local, V_thresh_local)
        V_reset_std  = std_p * abs(define_reasonable_V_reset(mean_p, E_L_est_local, V_thresh_local) - define_reasonable_V_reset(min(max(mean_p+1e-3,0.0),1.0), E_L_est_local, V_thresh_local)) / 1e-3
        return V_reset_mean, V_reset_std

    p_V_reset_map, tau_w_map, a_map, b_map = maps_s2
    V_reset_map = define_reasonable_V_reset(p_V_reset_map, E_L_est, V_thresh_est)
    V_reset_mean, V_reset_std = map_vreset_stats(means_s2[0], stds_s2[0], E_L_est, V_thresh_est)
    means_s2[0] = V_reset_mean
    stds_s2[0] = V_reset_std
    maps_s2[0] = V_reset_map
    
    print(f"Stage 2 MAP: V_reset = {V_reset_map:.2f} mV, tau_w = {tau_w_map:.2f} ms, a_w = {a_map:.2f} nS, b_w = {b_map:.3f} nA")
    
    # package results 
       
    means = np.concatenate([means_s1, means_s2])
    stds  = np.concatenate([stds_s1,  stds_s2])
    maps  = maps_s1 + maps_s2
    
    # prepare samples with correct V_reset column (map p_V_reset -> V_reset in mV)
    # convert tensors to numpy
    s1 = samples_s1.detach().cpu().numpy() if hasattr(samples_s1, "detach") else np.asarray(samples_s1)
    s2 = samples_s2.detach().cpu().numpy() if hasattr(samples_s2, "detach") else np.asarray(samples_s2)

    # s2 columns are [p_V_reset, tau_w, a, b]
    p = s2[:, 0]
    #V_thresh maps_s1[2]  # stage-1 MAP V_thresh used in stage-2
    V_reset_mV = (1.0 - p) * (E_L_est - 10.0) + p * (V_thresh_est - 1.0)

    # replace p_V_reset with V_reset (mV)
    s2[:, 0] = V_reset_mV

    # concatenate samples horizontally -> shape (N, 8)
    samples = np.concatenate([s1, s2], axis=1)

    return means, stds, maps, samples

def pack_experimental_and_simulate_SBI_voltage_trace_sequential(V_m_exp, I_inj_exp, spike_times_post_exp, E_L_est, V_thresh, tau_ref, SBI_parameters, T_window, t_vec_exp, dt_original, dt_target_V_m, dt_target_spike_train):
    # simulate SBI-inference based voltage trace & spike times & pack them with corresponding experimental arrays into a list
    
    # input
    # V_m_exp is the experimental membrane voltage in mV
    # I_inj_exp is the experimentally injected current in pA
    # spike_times_post_exp is the experimentally measured spike times in ms
    # E_L_est is the estimated resting potential in mV
    # V_thresh is the estimated spiking threshold in mV (for STA alignment on experimental data)
    # tau_ref is the absolute refractory period in ms
    # SBI_parameters are the SBI inferred parameters
    # T_window is the simulation time in ms
    # t_vec_exp is the experimental time vector in ms
    # dt_original is the resolution of the voltage trace in ms
    # dt_target_V_m is the target time step for the voltage treace in ms
    # dt_target_spike_train is the target time step for the spike times in ms
    
    # output
    # t_vec is a list of the experimental & SBI-based simulated time vectors in ms 
    # V_m_smooth_downsampled is a list of the experimental & SBI-based simulated membrane voltages in mV
    # spike_times_post is a list of the experimental & SBI-based simulated spike times in ms
    # r_post is a list of the experimental & SBI-based simulated firing rates in Hz
    
    # run simulation with SBI parameters
    t_vec_SBI_sim, V_m_SBI_sim, spike_times_post_SBI_sim = AdExp_I_inj(I_inj_exp*pA, SBI_parameters[1]*pF, SBI_parameters[0]*Mohm, E_L_est*mV, V_thresh*mV, SBI_parameters[3]*mV, SBI_parameters[2]*mV, SBI_parameters[4]*ms, SBI_parameters[5]*nS, SBI_parameters[6]*nA, tau_ref*ms)
                                                          #AdExp_I_inj(I_inj, C_m, R_m, E_L, V_thresh, V_reset, Delta_T, tau_w, a, b, tau_ref)
                                                          #maps=[R_m, C_m, Delta_T, V_reset, tau_w, a_w, b_w]

    # downsample experimental & simulated traces
    t_vec_exp_downsampled, time_exp_binarized, V_m_exp_subthreshold_smooth_downsampled, spike_train_exp_binarized, spike_train_counts_exp, spike_train_rates_exp = process_voltage_trace_spike_times_post(t_vec_exp, V_m_exp, spike_times_post_exp, V_thresh, dt_original, dt_target_V_m, dt_target_spike_train)
    t_vec_SBI_sim_downsampled, t_vec_SBI_sim_binarized, V_m_SBI_sim_smooth_downsampled, spike_train_SBI_sim_binarized, spike_train_counts_SBI_sim, spike_train_rates_SBI_sim = process_voltage_trace_spike_times_post(t_vec_SBI_sim, V_m_SBI_sim, spike_times_post_SBI_sim, V_thresh, dt_original, dt_target_V_m, dt_target_spike_train)

    # calculate firing rate
    r_post_exp = len(spike_times_post_exp) / T_window * 1000
    r_post_SBI_sim = len(spike_times_post_SBI_sim) / T_window * 1000
    
    return [t_vec_exp_downsampled, t_vec_SBI_sim_downsampled], [V_m_exp_subthreshold_smooth_downsampled, V_m_SBI_sim_smooth_downsampled], [spike_times_post_exp, spike_times_post_SBI_sim], [r_post_exp, r_post_SBI_sim]

######################## run full analysis function ########################

def process_single_Zeldenrust_file(file_name, data_folder, save_folder, sim_mode='small'): 
    # processes single Zeldenrust .mat data file
    # input 
    # file_name is the filename
    # data_folder is the relative path to the folder containing the raw .mat files
    # save_folder is the relative path to the folder to save the processed .pkl files
    # sim_mode determines if large or small numbers of SBI posteriors should be simulated

    # output 
    full_path = os.path.join(data_folder, file_name)
    print(f"Processing {file_name}") # track which trial is currently analyzed
    
    # load mat file
    mat = loadmat(full_path, struct_as_record=False, squeeze_me=True)
    #mat = loadmat_sanitized(full_path)
    matlab_data = mat.get('Data_acsf')

    if isinstance(matlab_data, np.ndarray):
        matlab_data_list = list(matlab_data)
        print(f"  → Detected {len(matlab_data_list)} records (array).")
    else:
        matlab_data_list = [matlab_data]
        print(f"  → Detected single ({len(matlab_data_list)}) record.")

    for idx, matlab_data_single in enumerate(matlab_data_list):
        try:
            data = load_data_experiment_Zeldenrust(matlab_data_single)
            if data is None:
                print(f"  Warning: Record {idx} is None — skipping.")
                continue

            # extract relevant data
            sampling_rate = data['sampling_rate_per_ms']
            hidden_state = downsample_decimate(data['hidden_state'], sampling_rate)  #data['hidden_state'][::int(sampling_rate*0.1)] # already downsampled by [::2] to always have 0.1 ms time step
            I_inj_exp = downsample_decimate(data['input_current_pA'], sampling_rate) #data['input_current_pA'][::int(sampling_rate*0.1)] # already downsampled by [::2] to always have 0.1 ms time step
            V_m_exp = downsample_decimate(data['membrane_voltage_mV'], sampling_rate) #data['membrane_voltage_mV'][::int(sampling_rate*0.1)] # already downsampled by [::2] to always have 0.1 ms time step
            spike_times_post_exp = data['spike_times_ms']
            firing_rate_exp = np.mean(data['firing_rate_Hz'])
            T_exp = data['duration_ms']
            t_vec_exp = np.linspace(0,T_exp,len(V_m_exp)) # create corresponding time array in ms
            dt = 0.1 # in ms simulation & sampling time step
            
            condition = data['condition']
            cell_type = data['cell_type']
            tau_switching = data['tau_switching_ms']
            
            V_thresh_FZ = np.mean(data['membrane_threshold_mV'])
            I_syn_mean = np.mean(data['input_current_pA'])
            MI_FZ = np.mean(data['MI_bits'])
            FI_FZ = np.mean(data['FI'])
            firing_rate_FZ_windowed = data['firing_rate_Hz']
            MI_FZ_windowed = data['MI_bits']
            FI_FZ_windowed = data['FI']
            nup_windowed = data['nup']
            ndown_windowed = data['ndown']
            nspikeperup_windowed = data['nspikeperup']
            nspikeperdown_windowed = data['nspikeperdown']
            
            base_name = os.path.splitext(file_name)[0]
            description = f"{base_name}_{idx}"
            
            # estimate E_L
            R_m_est, E_L_est, R2 = estimate_R_m_E_L_subthreshold(V_m_exp, I_inj_exp, V_thresh_FZ, plotting_mode=False)
            
            # run SBI to calculate tau_w
            
            # prepare experimental data
            V_m_exp_subthreshold = subthreshold_voltage_trace(V_m_exp, V_thresh_FZ)
            
            dt_original = dt
            T_window = 2000 # ms
            early_return_rate = 15 # Hz
            
            t_vec_window, V_m_exp_subthreshold_window, I_inj_exp_window, spike_times_post_exp_window, spike_times_post_exp_window_zeroed, best_start_idx, end_idx = extract_max_spikes_window(t_vec_exp, V_m_exp_subthreshold, I_inj_exp, spike_times_post_exp, dt_original=dt_original, T_window=T_window, early_return_rate=early_return_rate)

            # run SBI
            dt_target_V_m = 2.0 # ms
            dt_target_spike_train = 125.0 # ms
            pre_spike_time_window = 5.0 # ms
            post_spike_time_window = 20.0 # ms
            std_ms = 0.5 # ms
            tau_ref = 5 # ms
            
            """
            # 9D SBI
            num_simulations = 100000              # adjust for cluster (e.g., 60000+)
            savename = f"{description}_{sim_mode}_9D"      # single cache stem for the 9D run

            prior = build_prior_9D(E_L_est=E_L_est, V_thresh=V_thresh_FZ)
            
            #print(f"[RUN] Generating {num_simulations} simulations...")
            t0 = tp.time()
            theta, sims = run_simulations_9D(
                prior=prior,
                I_inj_exp=I_inj_exp_window,
                tau_ref=tau_ref,
                num_simulations=num_simulations,
                savename=savename)
            t1 = tp.time()
            
            print(f"[DONE] Simulations in {t1 - t0:.2f}s")
            
            print("[RUN] Training posterior and sampling...")
            
            means, stds, maps, samples = run_SBI_9D(
                V_m_exp=V_m_exp_subthreshold_window,
                I_inj_exp=I_inj_exp_window,
                spike_times_post_exp=spike_times_post_exp_window_zeroed,
                V_thresh=V_thresh_FZ,
                tau_ref=tau_ref,
                t_vec=t_vec_window,
                dt_original=dt_original,
                dt_target_V_m=dt_target_V_m,
                dt_target_spike_train=dt_target_spike_train,
                pre_spike_time_window=pre_spike_time_window,
                post_spike_time_window=post_spike_time_window,
                std_ms=std_ms,
                T_window=T_window,
                prior=prior,
                num_simulations=num_simulations,
                savename=savename,
                time_limit=1200)
            t2 = tp.time()
            print(f"[DONE] Full SBI in {t2 - t1:.2f}s (total {t2 - t0:.2f}s)")
            
            param_names=[r"$E_L$",r"$R_m$", r"$C_m$", r"$V_{thresh}$", r"$V_{reset}$", r"$Delta_T$", r"$\tau_w$", r"$a_w$", r"$b_w$"]
            param_units=["mV", "MOhm", "pF", "mV", "mV", "mV", "ms", "nS", "nA"]
            
            print("\n==== MAP ====")
            for name, val, un in zip(param_names, maps, param_units):
                print(f"{name:>12}: {val:.4f} {un:>12}")
            print("\n==== Mean ± SD ====")
            for name, mu, sd, un in zip(param_names, means, stds, param_units):
                print(f"{name:>12}: ({mu:.4f} ± {sd:.4f}) {un}")
                
            
            t_vecs, V_ms, spike_times_posts, r_posts = pack_experimental_and_simulate_SBI_voltage_trace_9D(V_m_exp_subthreshold_window, I_inj_exp_window, spike_times_post_exp_window_zeroed, V_thresh_FZ, tau_ref, means, T_window, t_vec_window, dt_original, dt_target_V_m, dt_target_spike_train)
            plot_cornerplot_with_voltage_trace(means, stds, maps, samples, t_vecs, V_ms, spike_times_posts, r_posts, param_names, param_units=param_units, description=description, savename=f"Exp_SBI_corner_plot_voltage_trace_{description}")


            """
            # sequential SBI
            if sim_mode=='small':
                num_simulations_s1=600
                num_simulations_s2=1200
            if sim_mode=='large':
                num_simulations_s1=6000
                num_simulations_s2=12000
            savename_s1=f"{description}_{sim_mode}_seq_s1" # os.path.join(sbi_log_folder, f"{description}_seq_s1") 
            savename_s2=f"{description}_{sim_mode}_seq_s2" #os.path.join(sbi_log_folder, f"{description}_seq_s2") 
            use_cached_s1=False
            use_cached_s2=False
            
            start = tp.time()
            
            means, stds, maps, samples = run_sbi_sequential(
                V_m_exp_subthreshold_window,
                I_inj_exp_window,
                spike_times_post_exp_window_zeroed,
                E_L_est,
                V_thresh_FZ,
                tau_ref, 
                t_vec_window,
                dt_original,
                dt_target_V_m, 
                dt_target_spike_train,
                pre_spike_time_window, 
                post_spike_time_window, 
                std_ms,
                T_window,
                num_simulations_s1,
                num_simulations_s2,
                savename_s1,
                savename_s2,
                use_cached_s1, 
                use_cached_s2)
            
            end = tp.time()
            print(f"Process took {end - start:.2f} seconds")
            
            print(f"Stage 0 estimates: V_thresh = {V_thresh_FZ:.2f} mV, E_L = {E_L_est:.2f} mV")

            # param naming for readability
            print(f"Stage 1 means: R_m = {means[0]:.2f} MOhm, C_m = {means[1]:.2f} pF, Delta_T = {means[2]:.2f} mV")
            print(f"Stage 1  MAPS:  R_m = {maps[0]:.2f} MOhm, C_m = {maps[1]:.2f} pF, Delta_T = {maps[2]:.2f} mV")
            
            print(f"Stage 2 means: V_reset = {means[3]:.2f} mV, tau_w = {means[4]:.2f} ms, a_w = {means[5]:.2f} nS, b_w = {means[6]:.3f} nA")
            print(f"Stage 2  MAPS:  V_reset = {maps[3]:.2f} mV, tau_w = {maps[4]:.2f} ms, a_w = {maps[5]:.2f} nS, b_w = {maps[6]:.3f} nA")
            
            # plot results
            param_names=[r"$R_m$", r"$C_m$", r"$Delta_T$", r"$V_{reset}$", r"$\tau_w$", r"$a_w$", r"$b_w$"]
            param_units=["MOhm", "pF", "mV", "mV", "ms", "nS", "nA"]
            t_vecs, V_ms, spike_times_posts, r_posts = pack_experimental_and_simulate_SBI_voltage_trace_sequential(V_m_exp_subthreshold_window, I_inj_exp_window, spike_times_post_exp_window_zeroed, E_L_est, V_thresh_FZ, tau_ref, means, T_window, t_vec_window, dt_original, dt_target_V_m, dt_target_spike_train)
            pf.plot_cornerplot_with_voltage_trace(means, stds, maps, samples, t_vecs, V_ms, spike_times_posts, r_posts, param_names, param_units, description, savename=f"Exp_SBI_corner_plot_voltage_trace_{description}_{sim_mode}") #os.path.join(fig_folder, f"Exp_SBI_corner_plot_voltage_trace_{description}"))
            
            
            # estimate MI
            neuron_type_mode_label = "UNK"  # default to avoid NameError
            if tau_switching == 250: # or tau_switching == 200:
                neuron_type_mode = 'excitatory'
                neuron_type_mode_label = 'EXC'
            
            if tau_switching == 50:
                neuron_type_mode = 'inhibitory'
                neuron_type_mode_label = 'INH'

            if tau_switching == 200:
                neuron_type_mode = 'Dopamine'
                neuron_type_mode_label = 'DOP'

            MI_I, MI_spike_train, MI_I_spike_train, H_xx, H_xy_input, H_xy_output, fraction_transferred_entropy, fraction_transferred_information = af.calculate_mutual_information_experimental_data_Zeldenrust(hidden_state, I_inj_exp, spike_times_post_exp, T_exp, neuron_type_mode=neuron_type_mode)
            MI_I_windowed, MI_spike_train_windowed, MI_I_spike_train_windowed, H_xx_windowed, H_xy_input_windowed, H_xy_output_windowed, fraction_transferred_entropy_windowed, fraction_transferred_information_windowed, firing_rate_Hz_windowed = af.calculate_mutual_information_windowed_experimental_data_Zeldenrust(hidden_state, I_inj_exp, spike_times_post_exp, T_exp, neuron_type_mode=neuron_type_mode)
            
            # change neuron_type_mode after information calculation because it does not matter for that but should be correct for saving
            if tau_switching == 250 and isinstance(cell_type, str): # filter out all fast spiking cells with inhibitory input (characterized by cell_type being a string)
                neuron_type_mode = 'inhibitory_with_excitatory_input'
                neuron_type_mode_label = 'INHwEXC'
            
            # calculate energy demand
            E_tot, energy_results = af.single_energy_calculation_experimental_data_Zeldenrust(means[0], means[1], E_L_est, V_thresh_FZ, firing_rate_exp, I_inj_exp/1000, T_exp) 
            
            print(f"File: {base_name}, Record: {idx}")
            print(f"  E_L: {E_L_est:.2f} mV")
            print(f"  V_thresh: {V_thresh_FZ:.2f} mV")
            print(f"  R_m (fit):     MAP {maps[0]:.2f} MOhm | mean {means[0]:.2f} ± {stds[0]:.2f}")
            print(f"  C_m (fit):     MAP {maps[1]:.2f} pF   | mean {means[1]:.2f} ± {stds[1]:.2f}")
            print(f"  ΔT (fit):       MAP {maps[2]:.2f} mV   | mean {means[2]:.2f} ± {stds[2]:.2f}")
            print(f"  V_reset (fit): MAP {maps[3]:.2f} mV   | mean {means[3]:.2f} ± {stds[3]:.2f}")
            print(f"  tau_w (fit):   MAP {maps[4]:.2f} ms   | mean {means[4]:.2f} ± {stds[4]:.2f}")
            print(f"  a_w (fit):     MAP {maps[5]:.2f} nS   | mean {means[5]:.2f} ± {stds[5]:.2f}")
            print(f"  b_w (fit):     MAP {maps[6]:.3f} nA   | mean {means[6]:.3f} ± {stds[6]:.3f}")
            print(f"  MI (spike train): {MI_spike_train:.2f} bits")
            print(f"  FI (fraction):    {fraction_transferred_information:.2f}")
            print(f"  tau_switching:    {tau_switching:.2f} ms")
            print(f"  cell type:        {cell_type}")
            print(f"  neuron type mode: {neuron_type_mode}")
            
            results_to_save = {
                'name': base_name,
                'index': idx,
                'neuron_type_mode': neuron_type_mode,
                'condition': condition,
                'cell_type': cell_type,
                'tau_switching_ms': float(tau_switching),
                'duration_ms': float(T_exp),
                'E_L_mV': float(E_L_est),
                'firing_rate_Hz': float(firing_rate_exp),
                'V_thresh_mV': float(V_thresh_FZ),
                'I_syn_mean': I_syn_mean,
            
                # SBI posterior summaries (maps/means/stds) — full 7 parameters
                'R_m_map_MOhm': float(maps[0]),
                'C_m_map_pF': float(maps[1]),
                'Delta_T_map_mV': float(maps[2]),
                'V_reset_map_mV': float(maps[3]),
                'tau_w_map_ms': float(maps[4]),
                'a_w_map_nS': float(maps[5]),
                'b_w_map_nA': float(maps[6]),
            
                'R_m_mean_MOhm': float(means[0]),
                'C_m_mean_pF': float(means[1]),
                'Delta_T_mean_mV': float(means[2]),
                'V_reset_mean_mV': float(means[3]),
                'tau_w_mean_ms': float(means[4]),
                'a_w_mean_nS': float(means[5]),
                'b_w_mean_nA': float(means[6]),
            
                'R_m_std_MOhm': float(stds[0]),
                'C_m_std_pF': float(stds[1]),
                'Delta_T_std_mV': float(stds[2]),
                'V_reset_std_mV': float(stds[3]),
                'tau_w_std_ms': float(stds[4]),
                'a_w_std_nS': float(stds[5]),
                'b_w_std_nA': float(stds[6]),
                
                # information metrics calculated 
                'MI_calculated_bits': float(MI_spike_train),
                'FI_calculated': float(fraction_transferred_information),
                'firing_rate_calculated_Hz_windowed': np.array(firing_rate_Hz_windowed), 
                'MI_calculated_bits_windowed': np.array(MI_spike_train_windowed),
                'FI_calculated_windowed': np.array(fraction_transferred_information_windowed),
                
                # information metrics Zeldenrust (FZ) 
                'MI_FZ_bits': float(MI_FZ),
                'FI_FZ': float(FI_FZ), 
                'firing_rate_FZ_Hz_windowed': np.array(firing_rate_FZ_windowed), 
                'MI_FZ_bits_windowed': np.array(MI_FZ_windowed),
                'FI_FZ_windowed': np.array(FI_FZ_windowed),
                'nup_windowed': np.array(nup_windowed),
                'ndown_windowed': np.array(ndown_windowed),
                'nspikeperup_windowed': np.array(nspikeperup_windowed),
                'nspikeperdown_windowed': np.array(nspikeperdown_windowed),
                
                # energy demand
                'E_tot_ATP_per_s': E_tot,
                'energy_results': energy_results,
                
                'MICE_calculated': MI_spike_train / firing_rate_exp,
                'MICE_FZ': MI_FZ / firing_rate_exp,
                'MI_calculated_per_energy': MI_spike_train / E_tot,
                'MI_FZ_per_energy': MI_FZ / E_tot,
                'MICE_calculated_per_energy': MI_spike_train / firing_rate_exp / E_tot,
                'MICE_FZ_per_energy': MI_FZ / firing_rate_exp / E_tot, 
                
                'MICE_calculated_windowed': np.array(MI_spike_train_windowed) / np.array(firing_rate_Hz_windowed),
                'MICE_FZ_windowed': np.array(MI_FZ_windowed) / np.array(firing_rate_FZ_windowed),
                'MI_calculated_per_energy_windowed': np.array(MI_spike_train_windowed) / E_tot,
                'MI_FZ_per_energy_windowed': np.array(MI_FZ_windowed) / E_tot,
                'MICE_calculated_per_energy_windowed': np.array(MI_spike_train_windowed) / np.array(firing_rate_Hz_windowed) / E_tot,
                'MICE_FZ_per_energy_windowed': np.array(MI_FZ_windowed) / np.array(firing_rate_FZ_windowed) / E_tot
                }
                        
            # create save name — add _idx if more than one record
            if len(matlab_data_list) == 1:
                pickle_name = os.path.join(save_folder, f"PS_{neuron_type_mode_label}_{base_name}_{sim_mode}.pkl")
            else:
                pickle_name = os.path.join(save_folder, f"PS_{neuron_type_mode_label}_{base_name}_{idx}_{sim_mode}.pkl")
                
            with open(os.path.join(save_folder, pickle_name), "wb") as f:
                pkl.dump(results_to_save, f)

            print(f"  Saved: {pickle_name}")
            
            #total_counter += 1
            #print(f"\nTotal number of data records processed: {total_counter}")
            

        except Exception as e:
            print(f"  Error processing record {idx} in file {file_name}: {e}")  


def simulate_V_m_SBI_posterior_samples(I_inj_exp, SBI_means, SBI_samples, E_L_est, V_thresh, tau_ref, n_samples=100):
    # simulate V_m for mean SBI parameters and for posterior samples

    # input
    # I_inj_exp is the experimental injected current in pA
    # SBI_means is a list of the mean SBI results ([R_m (Mohm), C_m (pF), V_reset (mV), Delta_T (mV), tau_w (ms), a (nS), b (nA)]
    # SBI_samples is an array of posterior samples ([R_m (Mohm), C_m (pF), V_reset (mV), Delta_T (mV), tau_w (ms), a (nS), b (nA)]
    # E_L_est is leak reversal potential in mV
    # V_thresh is the spiking threshold in mV
    # tau_ref is the refractory period in ms 
    # n_samples is the number of parameter samples to draw

    # output
    # t_vec_0_sim is the time vector in ms
    # V_m_mean is the V_m trace for mean parameters in mV
    # V_m_0_samples are the n_samples V_m traces for samples in mV
    # spike_times_post_0_samples are the spike times for samples in ms
    
    SBI_means = np.asarray(SBI_means, dtype=float)

    # central / mean parameters
    R_m_mean   = SBI_means[0]
    C_m_mean   = SBI_means[1]
    Delta_T_m  = SBI_means[2]
    V_reset_m  = SBI_means[3]
    tau_w_m    = SBI_means[4]
    a_m        = SBI_means[5]
    b_m        = SBI_means[6]

    t_vec_mean_sim, V_m_mean_sim, spike_times_post_mean_sim = AdExp_I_inj(I_inj_exp * pA, C_m_mean * pF, R_m_mean * Mohm, E_L_est * mV, V_thresh * mV, V_reset_m * mV, Delta_T_m * mV, tau_w_m * ms, a_m * nS, b_m * nA, tau_ref * ms)

    # convert to plain arrays
    T = len(V_m_mean_sim)

    SBI_samples = np.asarray(SBI_samples, dtype=float)
    N_total, n_params = SBI_samples.shape
    rng = np.random.default_rng()
    
    # choose n_samples indices without replacement
    n_samples = min(n_samples, N_total)
    idx = rng.choice(N_total, size=n_samples, replace=False)
    SBI_samples_sel = SBI_samples[idx]  # shape (n_samples, 7)

    # simulate for sampled parameter sets
    V_m_samples = np.empty((n_samples, T), dtype=float)

    spike_times_post_samples = []
    
    for k in range(n_samples):
        # draw one parameter vector from posterior samples
        R_m, C_m, Delta_T, V_reset, tau_w, a, b = SBI_samples_sel[k]

        t_vec_k, V_m_k, spikes_k = AdExp_I_inj(I_inj_exp * pA, C_m * pF, R_m * Mohm, E_L_est * mV, V_thresh * mV, V_reset * mV, Delta_T * mV, tau_w * ms, a * nS, b * nA, tau_ref * ms)

        V_m_samples[k, :] = np.asarray(V_m_k, dtype=float)
        spike_times_post_samples.append(spikes_k)
        

    return t_vec_mean_sim, V_m_mean_sim, V_m_samples, spike_times_post_mean_sim, spike_times_post_samples

       

######################## update functions ########################

def load_experimental_data_for_SBI_entry(data_SBI, data_exp_folder):
    # load experimental data (V_m_exp, I_inj_exp, spike_times_post_exp, t_vec_exp) corresponding to one analyzed SBI entry
    # input
    # data_SBI is the analyzed dictionary for one cell, containing at least 'name'  : base name of the analyzed file (e.g. 'NC_170725_aCSF_DopD2D1_E3_analyzed') and 'index' : integer index into the Data_acsf array in the .mat file
    # data_exp_folder is the path to the folder containing the raw .mat files
    
    # output
    # data_exp is a dictionary with entries
    
    # derive raw .mat file name from analyzed name
    # e.g. 'NC_170725_aCSF_DopD2D1_E3_analyzed' -> 'NC_170725_aCSF_DopD2D1_E3.mat'
    analyzed_name = data_SBI["name"]
    base_name = analyzed_name#.replace("_analyzed", "")
    file_name_mat = base_name + ".mat"

    full_path = os.path.join(data_exp_folder, file_name_mat)
    print(f"Loading raw data for {file_name_mat}")

    mat = loadmat(full_path, struct_as_record=False, squeeze_me=True)
    matlab_data = mat.get("Data_acsf")

    if isinstance(matlab_data, np.ndarray):
        matlab_data_list = list(matlab_data)
    else:
        matlab_data_list = [matlab_data]

    idx = int(data_SBI.get("index", 0))
    matlab_data_single = matlab_data_list[idx]

    # this reuses your existing loader
    data = load_data_experiment_Zeldenrust(matlab_data_single)

    sampling_rate = data["sampling_rate_per_ms"]

    # downsample using your helper (as in process_single_Zeldenrust_file)
    hidden_state_exp = downsample_decimate(data['hidden_state'], sampling_rate)  #data['hidden_state'][::int(sampling_rate*0.1)] # already downsampled by [::2] to always have 0.1 ms time step
    I_inj_exp = downsample_decimate(data['input_current_pA'], sampling_rate) #data['input_current_pA'][::int(sampling_rate*0.1)] # already downsampled by [::2] to always have 0.1 ms time step
    V_m_exp = downsample_decimate(data['membrane_voltage_mV'], sampling_rate) #data['membrane_voltage_mV'][::int(sampling_rate*0.1)] # already downsampled by [::2] to always have 0.1 ms time step
    spike_times_post_exp = data['spike_times_ms']
    firing_rate_exp = data['firing_rate_Hz']
    I_syn_mean = np.mean(data['input_current_pA'])
    
    T_exp = data["duration_ms"]
    t_vec_exp = np.linspace(0, T_exp, len(V_m_exp))  # time array in ms
    V_thresh_FZ = np.mean(data['membrane_threshold_mV'])
    
    # load information measures, not for SBI but further processing
    MI_FZ = np.mean(data['MI_bits'])
    FI_FZ = np.mean(data['FI'])
    firing_rate_FZ_windowed = data['firing_rate_Hz']
    MI_FZ_windowed = data['MI_bits']
    FI_FZ_windowed = data['FI']
    nup_windowed = data['nup']
    ndown_windowed = data['ndown']
    nspikeperup_windowed = data['nspikeperup']
    nspikeperdown_windowed = data['nspikeperdown']

    data_exp = {"t_vec_exp": t_vec_exp,  "V_m_exp": V_m_exp, "I_inj_exp": I_inj_exp, 'I_syn_mean': I_syn_mean, "spike_times_post_exp": spike_times_post_exp, "hidden_state_exp": hidden_state_exp, "T_exp": T_exp, "firing_rate_exp": firing_rate_exp, "membrane_threshold": V_thresh_FZ, 'MI_FZ_bits': MI_FZ, 'FI_FZ': FI_FZ, 'firing_rate_FZ_Hz_windowed': firing_rate_FZ_windowed,  'MI_FZ_bits_windowed': MI_FZ_windowed, 'FI_FZ_windowed': FI_FZ_windowed, 'nup_windowed': nup_windowed,'ndown_windowed': ndown_windowed, 'nspikeperup_windowed': nspikeperup_windowed, 'nspikeperdown_windowed': nspikeperdown_windowed}

    return data_exp

def update_calculate_mutual_information_windowed_experimental_data_Zeldenrust(data_exp_folder,data_analyzed_folder):
    # calculate windowed information for all analyzed files and update analyzed data files
    
    # input
    # data_exp_folder is the path to the folder containing raw .mat files
    # data_analyzed_folder is the path to the folder containing analyzed .pkl files


    # load analyzed data 
    analyzed_dict = load_analyzed_data(data_analyzed_folder)

    for key, analyzed_data in analyzed_dict.items():

        exp_dict = load_experimental_data_for_SBI_entry(analyzed_data, data_exp_folder)

        exp_data = exp_dict

        hidden_state = exp_data["hidden_state_exp"]
        I_inj = exp_data["I_inj_exp"]
        spike_times_post = exp_data["spike_times_post_exp"]
        T = exp_data.get("T_exp", len(hidden_state)) # in ms
        firing_rate = exp_data["firing_rate_exp"]

        neuron_type_mode = analyzed_data.get("neuron_type_mode", exp_data.get("neuron_type_mode"))
        
        MI_I, MI_spike_train, MI_I_spike_train, H_xx, H_xy_input, H_xy_output, fraction_transferred_entropy, fraction_transferred_information = af.calculate_mutual_information_experimental_data_Zeldenrust(hidden_state, I_inj, spike_times_post, T, neuron_type_mode=neuron_type_mode)
        MI_I_windowed, MI_spike_train_windowed, MI_I_spike_train_windowed, H_xx_windowed, H_xy_input_windowed, H_xy_output_windowed, fraction_transferred_entropy_windowed, fraction_transferred_information_windowed, firing_rate_Hz_windowed = af.calculate_mutual_information_windowed_experimental_data_Zeldenrust(hidden_state, I_inj, spike_times_post, T, neuron_type_mode=neuron_type_mode)
        
        # information metrics calculated 
        analyzed_data['MI_calculated_bits']= float(MI_spike_train)
        analyzed_data['FI_calculated']= float(fraction_transferred_information)
        analyzed_data['MICE_calculated']= MI_spike_train / firing_rate
        analyzed_data['firing_rate_calculated_Hz_windowed']= np.array(firing_rate_Hz_windowed)
        analyzed_data['MI_calculated_bits_windowed']= np.array(MI_spike_train_windowed)
        analyzed_data['FI_calculated_windowed']= np.array(fraction_transferred_information_windowed)
        analyzed_data['MICE_calculated_windowed']= np.array(MI_spike_train_windowed) / np.array(firing_rate_Hz_windowed)

        # information metrics Zeldenrust (FZ)
        analyzed_data['MI_FZ_bits']= exp_data['MI_FZ_bits']
        analyzed_data['FI_FZ']= exp_data['FI_FZ']
        analyzed_data['MICE_FZ']= np.mean(np.asarray(exp_data['MI_FZ_bits']) / firing_rate)
        analyzed_data['firing_rate_FZ_Hz_windowed']= exp_data['firing_rate_FZ_Hz_windowed']
        analyzed_data['MI_FZ_bits_windowed']= exp_data['MI_FZ_bits_windowed']
        analyzed_data['FI_FZ_windowed']= exp_data['FI_FZ_windowed']
        analyzed_data['MICE_FZ_windowed']= np.asarray(exp_data['MI_FZ_bits_windowed']) / np.asarray(exp_data['firing_rate_FZ_Hz_windowed'])
        analyzed_data['nup_windowed']= exp_data['nup_windowed']
        analyzed_data['ndown_windowed']= exp_data['ndown_windowed']
        analyzed_data['nspikeperup_windowed']= exp_data['nspikeperup_windowed']
        analyzed_data['nspikeperdown_windowed']= exp_data['nspikeperdown_windowed']

        #analyzed_data['information_windowed'] = window_results

        analyzed_path = os.path.join(data_analyzed_folder, key + ".pkl")
        with open(analyzed_path, "wb") as f:
            pkl.dump(analyzed_data, f)

        print(f"Updated '{key}.pkl' with windowed MI.")


def update_energy_calculation_experimental_data_Zeldenrust(data_exp_folder,data_analyzed_folder):
    # calculate E_tot of all analyzed files and update analyzed data files
    
    # input
    # data_exp_folder is the path to the folder containing raw .mat files
    # data_analyzed_folder is the path to the folder containing analyzed .pkl files

    # load analyzed data 
    analyzed_dict = load_analyzed_data(data_analyzed_folder)

    for key, analyzed_data in analyzed_dict.items():
        
        exp_data = load_experimental_data_for_SBI_entry(analyzed_data, data_exp_folder)

        I_inj_exp = exp_data["I_inj_exp"] # in pA
        spike_times_post = exp_data["spike_times_post_exp"] # in ms
        T_exp = exp_data["T_exp"] # in ms
        V_thresh_FZ = exp_data["membrane_threshold"] # in mV

        I_inj = I_inj_exp / 1000 # in nA
        T = T_exp / 1000 # in seconds
        r_post = len(spike_times_post) / T # in Hz
        
        R_m = analyzed_data['R_m_mean_MOhm']
        C_m = analyzed_data['C_m_mean_pF']
        E_L = analyzed_data['E_L_mV']

        E_tot, energy_results = af.single_energy_calculation_experimental_data_Zeldenrust(R_m, C_m, E_L, V_thresh_FZ, r_post, I_inj, T) 
        
        # load data 
        MI_calculated = analyzed_data['MI_calculated_bits']
        MICE_calculated = analyzed_data['MICE_calculated']
        MI_FZ = analyzed_data['MI_FZ_bits']
        MICE_FZ = analyzed_data['MICE_FZ']
        
        # save data
        analyzed_data['E_tot_ATP_per_s']= E_tot
        analyzed_data['energy_results']= energy_results
        
        analyzed_data['MI_calculated_per_energy']= MI_calculated / E_tot
        analyzed_data['MI_FZ_per_energy']= MI_FZ / E_tot
        analyzed_data['MICE_calculated_per_energy']= MICE_calculated / E_tot
        analyzed_data['MICE_FZ_per_energy']= MICE_FZ / E_tot 
        
        MI_calculated_bits_windowed = analyzed_data['MI_calculated_bits_windowed']
        MICE_calculated_windowed = analyzed_data['MICE_calculated_windowed']
        MI_FZ_bits_windowed = analyzed_data['MI_FZ_bits_windowed']
        MICE_FZ_windowed = analyzed_data['MICE_FZ_windowed']
        
        analyzed_data['MI_calculated_per_energy_windowed']= np.array(MI_calculated_bits_windowed) / E_tot
        analyzed_data['MI_FZ_per_energy_windowed']= np.array(MI_FZ_bits_windowed) / E_tot
        analyzed_data['MICE_calculated_per_energy_windowed']= np.array(MICE_calculated_windowed) / E_tot
        analyzed_data['MICE_FZ_per_energy_windowed']= np.array(MICE_FZ_windowed) / E_tot 
        

        analyzed_path = os.path.join(data_analyzed_folder, key + ".pkl")
        with open(analyzed_path, "wb") as f:
            pkl.dump(analyzed_data, f)

        print(f"Updated '{key}.pkl' with energy & chunked MI.")

def update_calculate_I_syn_experimental_data_Zeldenrust(data_exp_folder,data_analyzed_folder):
    # calculate windowed information for all analyzed files and update analyzed data files
    
    # input
    # data_exp_folder is the path to the folder containing raw .mat files
    # data_analyzed_folder is the path to the folder containing analyzed .pkl files


    # load analyzed data 
    analyzed_dict = load_analyzed_data(data_analyzed_folder)

    for key, analyzed_data in analyzed_dict.items():

        exp_dict = load_experimental_data_for_SBI_entry(analyzed_data, data_exp_folder)

        exp_data = exp_dict

        I_inj = exp_data["I_inj_exp"]
        
        analyzed_data['I_syn_mean'] = np.mean(I_inj)

        analyzed_path = os.path.join(data_analyzed_folder, key + ".pkl")
        with open(analyzed_path, "wb") as f:
            pkl.dump(analyzed_data, f)

        print(f"Updated '{key}.pkl' with I_syn_mean.")
        

######################## quality check functions ########################

def pack_experimental_and_simulate_SBI_voltage_trace_9D(V_m_exp, I_inj_exp, spike_times_post_exp, V_thresh, tau_ref, SBI_parameters, T_window, t_vec_exp, dt_original, dt_target_V_m, dt_target_spike_train):
    # simulate SBI-inference based voltage trace & spike times & pack them with corresponding experimental arrays into a list
    
    # input
    # V_m_exp is the experimental membrane voltage in mV
    # I_inj_exp is the experimentally injected current in pA
    # spike_times_post_exp is the experimentally measured spike times in ms
    ## E_L_est is the estimated resting potential in mV
    # V_thresh is the estimated spiking threshold in mV (for STA alignment on experimental data)
    # tau_ref is the absolute refractory period in ms
    # SBI_parameters are the SBI inferred parameters
    # T_window is the simulation time in ms
    # t_vec_exp is the experimental time vector in ms
    # dt_original is the resolution of the voltage trace in ms
    # dt_target_V_m is the target time step for the voltage treace in ms
    # dt_target_spike_train is the target time step for the spike times in ms
    
    # output
    # t_vec is a list of the experimental & SBI-based simulated time vectors in ms 
    # V_m_smooth_downsampled is a list of the experimental & SBI-based simulated membrane voltages in mV
    # spike_times_post is a list of the experimental & SBI-based simulated spike times in ms
    # r_post is a list of the experimental & SBI-based simulated firing rates in Hz
    
    # run simulation with SBI parameters

    # check where: 
    #V_reset_local = dafZ.define_reasonable_V_reset(SBI_parameters[4], SBI_parameters[0], SBI_parameters[3])
    #print(V_reset_local)
    
    t_vec_SBI_sim, V_m_SBI_sim, spike_times_post_SBI_sim = AdExp_I_inj(I_inj_exp*pA, SBI_parameters[2]*pF, SBI_parameters[1]*Mohm, SBI_parameters[0]*mV, SBI_parameters[3]*mV, SBI_parameters[4]*mV, SBI_parameters[5]*mV, SBI_parameters[6]*ms, SBI_parameters[7]*nS, SBI_parameters[8]*nA, tau_ref*ms)
                                                          #AdExp_I_inj(I_inj, C_m, R_m, E_L, V_thresh, V_reset, Delta_T, tau_w, a, b, tau_ref)
                                                          #maps=[E_L - 0, R_m  - 1, C_m  - 2, V_thresh  - 3, V_reset  - 4, Delta_T  - 5, tau_w  - 6, a_w  - 7, b_w  - 8]
    # downsample experimental & simulated traces
    t_vec_exp_downsampled, time_exp_binarized, V_m_exp_subthreshold_smooth_downsampled, spike_train_exp_binarized, spike_train_counts_exp, spike_train_rates_exp = process_voltage_trace_spike_times_post(t_vec_exp, V_m_exp, spike_times_post_exp, V_thresh, dt_original, dt_target_V_m, dt_target_spike_train)
    t_vec_SBI_sim_downsampled, t_vec_SBI_sim_binarized, V_m_SBI_sim_smooth_downsampled, spike_train_SBI_sim_binarized, spike_train_counts_SBI_sim, spike_train_rates_SBI_sim = process_voltage_trace_spike_times_post(t_vec_SBI_sim, V_m_SBI_sim, spike_times_post_SBI_sim, SBI_parameters[3], dt_original, dt_target_V_m, dt_target_spike_train)

    # calculate firing rate
    r_post_exp = len(spike_times_post_exp) / T_window * 1000
    r_post_SBI_sim = len(spike_times_post_SBI_sim) / T_window * 1000
    
    return [t_vec_exp_downsampled, t_vec_SBI_sim_downsampled], [V_m_exp_subthreshold_smooth_downsampled, V_m_SBI_sim_smooth_downsampled], [spike_times_post_exp, spike_times_post_SBI_sim], [r_post_exp, r_post_SBI_sim]


def spike_train_distance_measure_EMD(spike_times_post_exp, spike_times_post_sim):
    # compute the spike train distance between two spike trains based on the Earth Mover's Distance (EMD)
    
    # input
    # spike_times_post_exp is a 1D numpy array of spike times in ms for the experimental spike train
    # spike_times_post_sim is a 1D numpy array of spike times in ms for the simulated spike train
    
    # output
    # distance_EMD is a float giving the EMD spike train distance in ms (area between the two cumulative distribution functions)

    time_window_ms = 2000.0
    
    spike_times_exp = spike_times_post_exp #np.asarray(spike_times_post_exp, dtype=float).ravel()
    spike_times_sim = spike_times_post_sim #np.asarray(spike_times_post_sim, dtype=float).ravel()
    
    num_spikes_exp = spike_times_exp.size
    num_spikes_sim = spike_times_sim.size

    # handle special cases with empty spike trains
    if num_spikes_exp == 0 or num_spikes_sim == 0:
        return 0.0
    
    # general case: both trains non-empty
    # for 1D distributions the EMD can be computed as the integral over time of the absolute difference between their cumulative functions (CFs)
    
    # collect all time points where the cumulative functions can change.
    change_times = np.concatenate((np.array([0.0]), spike_times_exp, spike_times_sim, np.array([time_window_ms])))

    # sort and make unique to get ordered change points
    change_times = np.unique(change_times)

    # prepare counters for how many spikes have occurred up to a given time
    idx_exp = 0  # index in spike_times_exp
    idx_sim = 0  # index in spike_times_sim

    # initialize EMD value to accumulate
    distance_EMD = 0.0

    # loop over intervals [t_k, t_{k+1})
    for i in range(len(change_times) - 1):
        current_time = change_times[i]
        next_time = change_times[i + 1]
        interval_width = next_time - current_time

        # skip degenerate intervals just in case
        if interval_width <= 0.0:
            continue

        # update experimental spike count: count how many spikes occur at or before the current_time
        while idx_exp < num_spikes_exp and spike_times_exp[idx_exp] <= current_time:
            idx_exp += 1
        cumulative_mass_exp = idx_exp / float(num_spikes_exp)

        # update simulated spike count similarly
        while idx_sim < num_spikes_sim and spike_times_sim[idx_sim] <= current_time:
            idx_sim += 1
        cumulative_mass_sim = idx_sim / float(num_spikes_sim)

        # difference between the two cumulative functions on this interval
        cumulative_difference = abs(cumulative_mass_exp - cumulative_mass_sim)

        # contribution to the EMD is (difference in CF) * (length of interval)
        distance_EMD += cumulative_difference * interval_width

    return distance_EMD

def compute_means_vector_from_data(data_SBI):
    # build the parameter mean vector (length 9) in a fixed order from the data dictionary of one cell
    
    # input
    # data is a dictionary containing the fitted parameters for one cell
    
    # output
    # means is an array of length 9 with entries in this order [E_L, R_m, C_m, V_thresh, V_reset, Delta_T, tau_w, a_w, b_w]

    E_L_mV = data_SBI["E_L_mV"]
    R_m_mean_MOhm = data_SBI["R_m_mean_MOhm"]
    C_m_mean_pF = data_SBI["C_m_mean_pF"]
    V_thresh_mV = data_SBI["V_thresh_mV"]
    V_reset_mean_mV = data_SBI["V_reset_mean_mV"]
    Delta_T_mean_mV = data_SBI["Delta_T_mean_mV"]
    tau_w_mean_ms = data_SBI["tau_w_mean_ms"]
    a_w_mean_nS = data_SBI["a_w_mean_nS"]
    b_w_mean_nA = data_SBI["b_w_mean_nA"]

    means = np.array([E_L_mV, R_m_mean_MOhm, C_m_mean_pF, V_thresh_mV, V_reset_mean_mV, Delta_T_mean_mV, tau_w_mean_ms, a_w_mean_nS, b_w_mean_nA], dtype=float)

    return means


def compute_emd_for_one_cell(data_SBI, data_exp_folder):
    # compute EMD between experimental and simulated spike trains for one cell
    # input
    # data is a dictionary for one cell, 
    
    # output
    # EMD_value is the EMD spike train distance in ms

    # global variables
    T_window = 2000 # ms (2 s window, as requested)
    early_return_rate = 15.0 # Hz
    tau_ref = 5.0  # ms
    dt_target_V_m = 2.0  # ms
    dt_target_spike_train = 125.0 # ms
    dt_original = 0.1
    
    data_exp = load_experimental_data_for_SBI_entry(data_SBI, data_exp_folder)

    t_vec_exp = np.asarray(data_exp["t_vec_exp"], dtype=float)
    V_m_exp = np.asarray(data_exp["V_m_exp"], dtype=float)
    I_inj_exp = np.asarray(data_exp["I_inj_exp"], dtype=float)
    spike_times_post_exp = np.asarray(data_exp["spike_times_post_exp"], dtype=float)
    V_thresh_FZ = data_exp["membrane_threshold"] 
    #num_samples = int(T_window / dt_original)
    #t_vec_exp = np.linspace(0, T_window, num_samples) 
    
    # compute preprocessing
    V_m_exp_subthreshold = subthreshold_voltage_trace(V_m_exp, V_thresh_FZ)

    (t_vec_window, V_m_exp_subthreshold_window, I_inj_exp_window, spike_times_post_exp_window, spike_times_post_exp_window_zeroed, best_start_idx, end_idx) = extract_max_spikes_window(t_vec_exp, V_m_exp_subthreshold, I_inj_exp, spike_times_post_exp, dt_original=dt_original, T_window=T_window, early_return_rate=early_return_rate)

    # build means vector in correct order
    means = compute_means_vector_from_data(data_SBI)

    # run simulations
    t_vecs, V_ms, spike_times_posts, r_posts = (pack_experimental_and_simulate_SBI_voltage_trace_9D(V_m_exp_subthreshold_window, I_inj_exp_window, spike_times_post_exp_window_zeroed, V_thresh_FZ, tau_ref, means, T_window, t_vec_window, dt_original, dt_target_V_m, dt_target_spike_train))

    EMD_value = spike_train_distance_measure_EMD(np.asarray(spike_times_posts[0], dtype=float), np.asarray(spike_times_posts[1], dtype=float),)
    print(EMD_value)
    return EMD_value


def compute_emd_for_all_files_and_update_csv(data_exp_folder, data_SBI_folder):
    # description of function task
    # for each analyzed .pkl file in data_folder:
    #   1) load the data via dafZ.load_analyzed_data
    #   2) compute the EMD between experimental and simulated spike trains
    #   3) write the EMD into the "quality_EMD" column of file_quality_table.csv
    
    # input
    # data_exp_folder is the path to the folder containing raw .mat files
    # data_SBI_folder is the path to the folder containing analyzed .pkl files and the file file_quality_table.csv
    
    # output
    # df_updated is the pandas DataFrame of the updated quality table

    # load all analyzed data
    results_dict = load_analyzed_data(data_SBI_folder)

    # dictionary to store EMD per analyzed key
    emd_per_key = {}

    # loop over all result entries and compute EMD
    for key, data_SBI in results_dict.items():
        try:
            emd_value = compute_emd_for_one_cell(data_SBI, data_exp_folder)
            emd_per_key[key] = emd_value
            print(f"Computed EMD for {key}: {emd_value:.4f} ms")
        except Exception as e:
            print(f"[WARNING] Could not compute EMD for {key}: {repr(e)}")
            continue

    # load file_quality_table.csv and update "quality_EMD" column
    quality_table_path = os.path.join(data_SBI_folder, "file_quality_table.csv")

    df = pd.read_csv(quality_table_path)

    # ensure the column exists
    if "quality_EMD" not in df.columns:
        df["quality_EMD"] = np.nan

    name_without_ext = df["name"].apply(lambda x: os.path.splitext(str(x))[0])

    for i, base_name in enumerate(name_without_ext):
        if base_name in emd_per_key:
            df.at[i, "quality_EMD"] = emd_per_key[base_name]

    # save updated CSV back to disk
    df.to_csv(quality_table_path, index=False)
    print(f"Updated file_quality_table.csv with EMD values at {quality_table_path}")

    return df

def filter_quality_checked(results_dict_raw, sequential_SBI_name, sequential_SBI_quality_final):
    # filters raw SBI results using quality information
    
    # input
    # results_dict_raw is a dictionary where keys are filenames and values are result objects
    # sequential_SBI_name is a list of filenames (strings) corresponding to analyzed data
    # sequential_SBI_quality_final is a list with the same length, containing 'Y' or 'N'

    # output
    # results_dict_quality_checked is a filtered dictionary containing only entries
    # whose filename appears in sequential_SBI_name and has quality 'Y'

    # build quality lookup table
    quality_lookup = {}
    for name, q in zip(sequential_SBI_name, sequential_SBI_quality_final):
        fname = os.path.basename(name) # keep only basename
        if fname.endswith(".pkl"): # strip ".pkl"
            fname = fname[:-4]
        quality_lookup[fname] = q

    # filter results_dict_raw
    results_dict_quality_checked = {}
    for key, value in results_dict_raw.items():
        k_fname = os.path.basename(key)
        if k_fname.endswith(".pkl"):
            k_fname = k_fname[:-4]

        if quality_lookup.get(k_fname) == "Y":
            results_dict_quality_checked[key] = value

    return results_dict_quality_checked

######################## loading results functions ########################

def load_analyzed_data(data_folder): 
    # collect all analyzed .pkl files
    # input
    # data_folder is the path to folder containing result .pkl files

    # output
    # results_dict is the dictionary containing all results 
    
    results_dict = {}
    
    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".pkl") or not file_name.startswith("PS_"):
            continue
    
        full_path = os.path.join(data_folder, file_name)
        with open(full_path, "rb") as f:
            data = pkl.load(f)
    
        # create a key based on file name without extension
        key = os.path.splitext(file_name)[0]
        results_dict[key] = data
    
    print(f"Loaded {len(results_dict)} result files.")

    return results_dict

def extract_results_by_mode(data_folder, mode='all', sequential_SBI_name=None, sequential_SBI_quality_final=None):
    # load and extract relevant features from Zeldenrust analyzed data
    
    # input
    # data_folder is the path to folder containing result .pkl files
    # mode is the condition of which cells to exctract with option 'all', 'excitatory', 'inhibitory', 'inhibitory_with_excitatory_input', 'Dopamine'
    # sequential_SBI_name is a list of filenames and enables quality check
    # sequential_SBI_quality_final is a list with 'Y'/'N'
    
    # output
    # results_analysis is a dictionary of lists of results
    
    # quality check if given
    quality_lookup = None
    if sequential_SBI_name is not None and sequential_SBI_quality_final is not None:
        quality_lookup = {}
        for name, q in zip(sequential_SBI_name, sequential_SBI_quality_final):
            if not isinstance(name, str):
                continue
            fname = os.path.basename(name)
            if fname.endswith(".pkl"):
                fname = fname[:-4]
            quality_lookup[fname] = q
            
    # initialize output lists
    name_list = []
    index_list = []
    neuron_type_mode_list = []
    condition_list = []
    cell_type_list = []
    tau_switching_ms_list = []
    duration_ms_list = []
    E_L_mV_list = []
    firing_rate_Hz_list = []
    V_thresh_mV_list = []
    I_syn_mean_list = []

    # MAPs
    R_m_map_MOhm_list = []
    C_m_map_pF_list = []
    Delta_T_map_mV_list = []
    V_reset_map_mV_list = []
    tau_w_map_ms_list = []
    a_w_map_nS_list = []
    b_w_map_nA_list = []

    # means
    R_m_mean_MOhm_list = []
    C_m_mean_pF_list = []
    Delta_T_mean_mV_list = []
    V_reset_mean_mV_list = []
    tau_w_mean_ms_list = []
    a_w_mean_nS_list = []
    b_w_mean_nA_list = []

    # STDs
    R_m_std_MOhm_list = []
    C_m_std_pF_list = []
    Delta_T_std_mV_list = []
    V_reset_std_mV_list = []
    tau_w_std_ms_list = []
    a_w_std_nS_list = []
    b_w_std_nA_list = []

    # information metrics
    MI_calculated_bits_list = []
    MI_FZ_bits_list = []
    FI_calculated_list = []
    FI_FZ_list = []
    
    firing_rate_calculated_Hz_windowed = []
    MI_calculated_bits_windowed_list = []
    MI_FZ_bits_windowed_list = []
    FI_calculated_windowed_list = []
    FI_FZ_windowed_list = []
    nup_windowed_list = []
    ndown_windowed_list = []
    nspikeperup_windowed_list = []
    nspikeperdown_windowed_list = []
    
    # energy measures
    E_tot_1e9_ATP_per_s_list = []
    energy_results_list = []
    
    # efficiency measures
    MICE_calculated_list = []
    MICE_FZ_list = []
    MI_calculated_per_energy_list = []
    MI_FZ_per_energy_list = []
    MICE_calculated_per_energy_list = []
    MICE_FZ_per_energy_list = []
    
    MICE_calculated_windowed_list = [] 
    MICE_FZ_windowed_list = [] 
    MI_calculated_per_energy_windowed_list = [] 
    MI_FZ_per_energy_windowed_list = [] 
    MICE_calculated_per_energy_windowed_list = [] 
    MICE_FZ_per_energy_windowed_list = [] 

    # load all desired .pkl results
    for file_name in os.listdir(data_folder):
        if not file_name.endswith(".pkl") or not file_name.startswith("PS_"):
            continue
        
        if quality_lookup is not None:
            base = file_name[:-4]  # strip .pkl
            if quality_lookup.get(base) != "Y":
                continue

        full_path = os.path.join(data_folder, file_name)
        try:
            with open(full_path, "rb") as f:
                res = pkl.load(f)
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            continue

        # get neuron_type_mode from data
        neuron_type_mode = res.get('neuron_type_mode', 'unknown')

        # filter by mode (and exclude Dopamine when mode == 'all', to match previous behavior)
        if mode != 'all' and neuron_type_mode != mode:
            continue
        if mode == 'all' and neuron_type_mode == 'Dopamine':
            continue

        try:
            # Basics
            name_list.append(res.get('name'))
            index_list.append(res.get('index'))
            neuron_type_mode_list.append(neuron_type_mode)
            condition_list.append(res.get('condition'))
            cell_type_list.append(res.get('cell_type'))
            tau_switching_ms_list.append(res.get('tau_switching_ms'))
            duration_ms_list.append(res.get('duration_ms'))
            E_L_mV_list.append(res.get('E_L_mV'))
            firing_rate_Hz_list.append(res.get('firing_rate_Hz'))
            V_thresh_mV_list.append(res.get('V_thresh_mV'))
            I_syn_mean_list.append(res.get('I_syn_mean'))

            # MAPs
            R_m_map_MOhm_list.append(res.get('R_m_map_MOhm'))
            C_m_map_pF_list.append(res.get('C_m_map_pF'))
            Delta_T_map_mV_list.append(res.get('Delta_T_map_mV'))
            V_reset_map_mV_list.append(res.get('V_reset_map_mV'))
            tau_w_map_ms_list.append(res.get('tau_w_map_ms'))
            a_w_map_nS_list.append(res.get('a_w_map_nS'))
            b_w_map_nA_list.append(res.get('b_w_map_nA'))

            # Means
            R_m_mean_MOhm_list.append(res.get('R_m_mean_MOhm'))
            C_m_mean_pF_list.append(res.get('C_m_mean_pF'))
            Delta_T_mean_mV_list.append(res.get('Delta_T_mean_mV'))
            V_reset_mean_mV_list.append(res.get('V_reset_mean_mV'))
            tau_w_mean_ms_list.append(res.get('tau_w_mean_ms'))
            a_w_mean_nS_list.append(res.get('a_w_mean_nS'))
            b_w_mean_nA_list.append(res.get('b_w_mean_nA'))

            # STDs
            R_m_std_MOhm_list.append(res.get('R_m_std_MOhm'))
            C_m_std_pF_list.append(res.get('C_m_std_pF'))
            Delta_T_std_mV_list.append(res.get('Delta_T_std_mV'))
            V_reset_std_mV_list.append(res.get('V_reset_std_mV'))
            tau_w_std_ms_list.append(res.get('tau_w_std_ms'))
            a_w_std_nS_list.append(res.get('a_w_std_nS'))
            b_w_std_nA_list.append(res.get('b_w_std_nA'))

            # Information metrics
            MI_calculated_bits_list.append(res.get('MI_calculated_bits'))
            MI_FZ_bits_list.append(res.get('MI_FZ_bits'))
            FI_calculated_list.append(res.get('FI_calculated'))
            FI_FZ_list.append(res.get('FI_FZ'))
            #info_win = res.get('information_windowed')
            
            firing_rate_calculated_Hz_windowed.append(res.get('firing_rate_calculated_Hz_windowed'))
            MI_calculated_bits_windowed_list.append(res.get('MI_calculated_bits_windowed'))
            FI_calculated_windowed_list.append(res.get('FI_calculated_windowed'))
            MI_FZ_bits_windowed_list.append(res.get('MI_FZ_bits_windowed'))
            FI_FZ_windowed_list.append(res.get('FI_FZ_windowed'))
            
            nup_windowed_list.append(res.get('nup_windowed'))
            ndown_windowed_list.append(res.get('ndown_windowed'))
            nspikeperup_windowed_list.append(res.get('nspikeperup_windowed'))
            nspikeperdown_windowed_list.append(res.get('nspikeperdown_windowed'))
            
            E_tot_1e9_ATP_per_s_list.append(res.get('E_tot_ATP_per_s'))
            energy_results_list.append(res.get('energy_results'))
            
            MICE_calculated_list.append(res.get('MICE_calculated'))
            MICE_FZ_list.append(res.get('MICE_FZ'))
            MI_calculated_per_energy_list.append(res.get('MI_calculated_per_energy'))
            MI_FZ_per_energy_list.append(res.get('MI_FZ_per_energy'))
            MICE_calculated_per_energy_list.append(res.get('MICE_calculated_per_energy'))
            MICE_FZ_per_energy_list.append(res.get('MICE_FZ_per_energy'))
            
            MICE_calculated_windowed_list.append(res.get('MICE_calculated_windowed')) 
            MICE_FZ_windowed_list.append(res.get('MICE_FZ_windowed'))
            MI_calculated_per_energy_windowed_list.append(res.get('MI_calculated_per_energy_windowed'))
            MI_FZ_per_energy_windowed_list.append(res.get('MI_FZ_per_energy_windowed'))
            MICE_calculated_per_energy_windowed_list.append(res.get('MICE_calculated_per_energy_windowed')) 
            MICE_FZ_per_energy_windowed_list.append(res.get('MICE_FZ_per_energy_windowed'))
            
            
        except Exception as e:
            print(f"Error extracting from {file_name}: {e}")
            continue

    results_analysis = {
        # basics
        'name_list': name_list,
        'index_list': index_list,
        'neuron_type_mode_list': neuron_type_mode_list,
        'condition_list': condition_list,
        'cell_type_list': cell_type_list,
        'tau_switching_ms_list': tau_switching_ms_list,
        'duration_ms_list': duration_ms_list,
        'E_L_mV_list': E_L_mV_list,
        'firing_rate_Hz_list': firing_rate_Hz_list,
        'V_thresh_mV_list': V_thresh_mV_list,
        'I_syn_mean_pA_list': I_syn_mean_list,

        # MAPs
        'R_m_map_MOhm_list': R_m_map_MOhm_list,
        'C_m_map_pF_list': C_m_map_pF_list,
        'Delta_T_map_mV_list': Delta_T_map_mV_list,
        'V_reset_map_mV_list': V_reset_map_mV_list,
        'tau_w_map_ms_list': tau_w_map_ms_list,
        'a_w_map_nS_list': a_w_map_nS_list,
        'b_w_map_nA_list': b_w_map_nA_list,

        # Means
        'R_m_mean_MOhm_list': R_m_mean_MOhm_list,
        'C_m_mean_pF_list': C_m_mean_pF_list,
        'Delta_T_mean_mV_list': Delta_T_mean_mV_list,
        'V_reset_mean_mV_list': V_reset_mean_mV_list,
        'tau_w_mean_ms_list': tau_w_mean_ms_list,
        'a_w_mean_nS_list': a_w_mean_nS_list,
        'b_w_mean_nA_list': b_w_mean_nA_list,

        # STDs
        'R_m_std_MOhm_list': R_m_std_MOhm_list,
        'C_m_std_pF_list': C_m_std_pF_list,
        'Delta_T_std_mV_list': Delta_T_std_mV_list,
        'V_reset_std_mV_list': V_reset_std_mV_list,
        'tau_w_std_ms_list': tau_w_std_ms_list,
        'a_w_std_nS_list': a_w_std_nS_list,
        'b_w_std_nA_list': b_w_std_nA_list,

        # Information metrics
        'MI_calculated_bits_list': MI_calculated_bits_list,
        'MI_FZ_bits_list': MI_FZ_bits_list,
        'FI_calculated_list': FI_calculated_list,
        'FI_FZ_list': FI_FZ_list, 
        'MI_calculated_bits_windowed_list': MI_calculated_bits_windowed_list,
        'MI_FZ_bits_windowed_list': MI_FZ_bits_windowed_list,
        
        'firing_rate_calculated_Hz_windowed': firing_rate_calculated_Hz_windowed, 
        'FI_calculated_windowed_list': FI_calculated_windowed_list,
        'FI_FZ_windowed_list': FI_FZ_windowed_list,
        'nup_windowed_list': nup_windowed_list, 
        'ndown_windowed_list': ndown_windowed_list,
        'nspikeperup_windowed_list': nspikeperup_windowed_list,
        'nspikeperdown_windowed_list': nspikeperdown_windowed_list,
        

        'E_tot_1e9_ATP_per_s_list': E_tot_1e9_ATP_per_s_list,
        'energy_results_list': energy_results_list, 
        
        'MICE_calculated_list': MICE_calculated_list,
        'MICE_FZ_list': MICE_FZ_list,
        'MI_calculated_per_energy_list' : MI_calculated_per_energy_list,
        'MI_FZ_per_energy_list' : MI_FZ_per_energy_list,
        'MICE_calculated_per_energy_list' : MICE_calculated_per_energy_list,
        'MICE_FZ_per_energy_list' : MICE_FZ_per_energy_list,
        
        'MICE_calculated_windowed_list': MICE_calculated_windowed_list, 
        'MICE_FZ_windowed_list': MICE_FZ_windowed_list, 
        'MI_calculated_per_energy_windowed_list': MI_calculated_per_energy_windowed_list, 
        'MI_FZ_per_energy_windowed_list': MI_FZ_per_energy_windowed_list, 
        'MICE_calculated_per_energy_windowed_list': MICE_calculated_per_energy_windowed_list, 
        'MICE_FZ_per_energy_windowed_list': MICE_FZ_per_energy_windowed_list}

    return results_analysis

 
######################## SBI plotting functions ########################


def plot_marginal_posteriors(maps, means, stds, samples, description, savename=None):
    # plots marginal posteriors
    
    # input
    # maps are the maximum a posteriori estimated values 
    # means are the mean estimated values 
    # stds are the standard deviations of the estimated values
    # samples are samples of the posterior distribution
    # description is the title of the figure
    # savename is an optional name to save figure
    
    param_names = [r"$V_{thresh}$ / mV", r"$\tau_w$ / ms", r"$C_m$ / pF"]
    param_units = ["mV", "ms", "pF"]
    
    fig, axs = plt.subplots(1, len(maps), figsize=(10,5))
    axs = axs.flatten()
    
    for i in range(len(maps)):
        samples_i = samples[:, i].numpy()
        map_i = maps[i]
        mean_i = means[i]
        std_i = stds[i]
        param_name_i = param_names[i]
        param_units_i = param_units[i]
    
        sns.kdeplot(samples_i, fill=True, ax=axs[i])
        axs[i].axvline(map_i, color="red", linestyle="--", label=f"MAP: {map_i:.1f} {param_units_i}")
        axs[i].axvline(mean_i, color="red", linestyle="-.", label=f"Mean: {mean_i:.1f} ± {std_i:.1f} {param_units_i}")
        axs[i].set_xlabel(param_name_i)
        axs[i].set_ylabel("Density")
        axs[i].legend()
    
    plt.suptitle(f"Posterior distributions of AdExp parameters for {description}", fontsize=16)
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig('../Figures/' + str(savename)+'.pdf')
    
    plt.show()

def plot_joint_posteriors(maps, means, stds, samples, description, savename=None):
    # plots joint posteriors
    
    # input
    # maps are the maximum a posteriori estimated values 
    # means are the mean estimated values 
    # stds are the standard deviations of the estimated values
    # samples are samples of the posterior distribution
    # description is the title of the figure
    # savename is an optional name to save figure

    # convert samples to numpy if it's a tensor
    samples_np = samples.numpy()
    
    # create a joint 2D KDE plot
    plt.figure(figsize=(7, 6))
    sns.kdeplot(
        x=samples_np[:, 0],  # V_thresh
        y=samples_np[:, 1],  # tau_w
        fill=True,
        cmap="viridis",
        thresh=0.05,
        levels=100)
    
    # add map & mean estimate point
    plt.scatter(maps[0], maps[1], color='red', label="MAP", zorder=5)
    plt.scatter(means[0], means[1], color='orange', label="Mean", zorder=5)
    plt.xlabel(r"$V_{thresh}$ / mV")
    plt.ylabel(r"$\tau_w$ / ms")
    plt.title(f"Joint posterior density of {description}")
    plt.legend()
    plt.tight_layout()
    
    if savename is not None:
        plt.savefig('../Figures/' + str(savename)+'1.pdf')
    
    plt.show()
    

################## PCA analysis functions ########################


def build_feature_matrix(results_exc, results_inh, features_list):
    # build a combined feature matrix for exc + inh cells
    # input
    # results_exc, results_inh are the dictionaries of excitatory & inhibitory cell results
    # features_list is a list of feature names
    
    # output
    # X_full is a feature matrix ((n_samples, n_features)-array) with exc followed by inh cells
    # cell_types_list is an array of cell type strings ('exc' or 'inh')
    
    # build exc matrix
    exc_features = []
    for feature_name in features_list:
        arr = np.asarray(results_exc[feature_name]).ravel()
        exc_features.append(arr)
    X_exc = np.column_stack(exc_features)

    # build inh matrix
    inh_features = []
    for feature_name in features_list:
        arr = np.asarray(results_inh[feature_name]).ravel()
        inh_features.append(arr)
    X_inh = np.column_stack(inh_features)

    # stack: exc first, then inh
    X_full = np.vstack([X_exc, X_inh])

    # labels
    cell_types_list = np.array(['exc'] * X_exc.shape[0] + ['inh'] * X_inh.shape[0])

    return X_full, cell_types_list

def run_pca_analyzed_data_Zeldenrust(results_exc, results_inh, features_list, n_components=3):
    # compute PCA on combined exc & inh data
    # input
    # results_exc, results_inh are the dictionaries of excitatory & inhibitory cell results
    # features_list is a list of feature names
    # n_components is the number of PCs to return
    
    # output
    # pca is the fitted PCA object
    # scores are the PCA scores (PC coordinates) for each cell (n_samples, n_components)
    # cell_types_list is an array of cell type strings ('exc' or 'inh')    

    # build feature matrix
    X_full, cell_types_list = build_feature_matrix(results_exc, results_inh, features_list)
    
    # standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(X_full)

    # run PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_full_scaled)

    return pca, scores, cell_types_list, X_full_scaled, scaler

def build_dic_exp_data(results_analysis, pca, scores, X_full_scaled, scaler):
    # create a dictionary of dictionaries from results_analysis (dict of lists) and add PCA-related values
    #
    # input
    # results_analysis is a dictionary of lists (all lists same length, one entry per cell)
    # pca is the fitted PCA object
    # scores is an array (N, n_components) with PCA scores for each cell
    # X_full_scaled is an array (N, n_features) with standardized features for each cell (input to PCA)
    # scaler is the fitted StandardScaler used to create X_full_scaled
    #
    # output
    # dic_exp is a dictionary of dictionaries:
    # dic_exp[cell_key] = { all per-cell values from results_analysis, plus 'pca', 'scores', 'X_full_scaled', 'scaler' }

    import numpy as np

    # convert arrays
    scores = np.asarray(scores)
    X_full_scaled = np.asarray(X_full_scaled)

    # determine N from a reliable list
    n = len(results_analysis['neuron_type_mode_list'])
    
    # sanity checks
    if scores.shape[0] != n:
        raise ValueError(f"scores has {scores.shape[0]} rows but results_analysis has {n} entries.")
    if X_full_scaled.shape[0] != n:
        raise ValueError(f"X_full_scaled has {X_full_scaled.shape[0]} rows but results_analysis has {n} entries.")

    # determine which keys are per-cell lists of length n
    list_keys = []
    for k, v in results_analysis.items():
        if isinstance(v, (list, np.ndarray)) and len(v) == n:
            list_keys.append(k)

    # build dict-of-dicts
    dic_exp = {}
    cell_counter = {}

    for i in range(n):
        # get neuron type string to build key
        if 'neuron_type_mode_list' in results_analysis:
            nt = str(results_analysis['neuron_type_mode_list'][i])
        else:
            nt = 'celltype'

        # count per neuron type
        if nt not in cell_counter:
            cell_counter[nt] = 0
        cell_counter[nt] += 1

        # key format: "{neuron_type}_cell{count}"
        cell_key = f"{nt}_cell_{cell_counter[nt]}"

        # entry: all per-cell list values
        entry = {k: results_analysis[k][i] for k in list_keys}

        # add PCA-related values for this cell
        entry['pca'] = pca
        entry['scores'] = scores[i, :].copy()
        entry['X_full_scaled'] = X_full_scaled[i, :].copy()
        entry['scaler'] = scaler
        # add explicit PC coordinates
        entry['pc1'] = float(scores[i, 0])
        entry['pc2'] = float(scores[i, 1])
        entry['pc3'] = float(scores[i, 2])


        dic_exp[cell_key] = entry

    return dic_exp







































######################## OLD 9D SBI functions ########################


def OLD_downsample_mean(x, sampling_rate):
    # simple mean-based downsampling
    # input
    # x is a 1D array
    # sampling_rate is the (integer) sampling rate of the experimental data

    # output
    # x_downsampled is a 1D array of the downsampled x
    
    factor = int(sampling_rate * 0.1) # with 0.1 being the target sampling rate for x_downsampled
    n = len(x) // factor
    x = x[:n*factor]  # trim excess
    x_downsampled = x.reshape(n, factor).mean(axis=1)
    return x_downsampled



def simulator_wrapper_9D(params_tensor, I_inj_exp, tau_ref):
    # simulator wrapper for SBI inference

    # input
    # params_tensor is a torch tensor
    # I_inj_exp is the experimentally injected current in pA
    # tau_ref is the absolute refractory period in ms , tau_ref
    
    # output
    # sim_tensor is the feature tensor
    
    # params order (9D):
    # 0) E_L / mV
    # 1) R_m / MOhm
    # 2) C_m / pF
    # 3) V_thresh / mV
    # 4) p_V_reset [0 -- 1] -> mapped to [E_L-10 .. V_thresh-1] mV
    # 5) Delta_T / mV
    # 6) tau_w / ms
    # 7) a / nS
    # 8) b / nA
    
    E_L_b      = params_tensor[0].item() * mV
    R_m_b      = params_tensor[1].item() * Mohm
    C_m_b      = params_tensor[2].item() * pF
    V_thresh_b = params_tensor[3].item() * mV
    p_V_reset  = params_tensor[4].item()
    V_reset_b  = define_reasonable_V_reset(p_V_reset, E_L_b/mV, V_thresh_b/mV) * mV
    Delta_T_b  = params_tensor[5].item() * mV
    tau_w_b    = params_tensor[6].item() * ms
    a_b        = params_tensor[7].item() * nS
    b_b        = params_tensor[8].item() * nA

    I_inj_exp_b = I_inj_exp * pA
    tau_ref_b = tau_ref*ms

    # run model
    time_sim, V_m_sim, spike_times_post_sim = AdExp_I_inj(I_inj_exp_b, C_m_b, R_m_b, E_L_b, V_thresh_b, V_reset_b, Delta_T_b, tau_w_b, a_b, b_b, tau_ref_b)

    # compare to experimental: downsample + rates using your existing helper, important: pass "V_thresh_b/mV" as scalar for threshold to the processing
    #_, _, V_m_sim_smooth_down, spike_train_sim_bin, _, spike_rates_sim = process_voltage_trace_spike_times_post(time_sim, V_m_sim, spike_times_post_sim, V_thresh=V_thresh_b/mV, dt_original=dt_original, dt_target_V_m=1, dt_target_spike_train=1)

    V_m_sim_tensor       = torch.tensor(np.asarray(V_m_sim), dtype=torch.float32)
    spike_times_post_sim_tensor = torch.tensor(np.asarray(spike_times_post_sim), dtype=torch.float32)
    
    sim_tensor = torch.cat([V_m_sim_tensor, spike_times_post_sim_tensor], dim=0)
    
    return sim_tensor


def build_prior_9D(E_L_est, V_thresh):
    # define prior for 9D vector [E_L, R_m, C_m, V_thresh, p_V_reset, Delta_T, tau_w, a, b]
    # input
    # E_L_est is the estimated resting potential in mV
    # V_thresh is the estimated spiking threshold in mV (for STA alignment on experimental data)   
    
    # output
    # priors are the prior boundaries prepared for SBI
        
    
    low  = torch.tensor([
        E_L_est - 10.0,   # E_L / mV
        25.0,             # R_m / mOhm
        1.0,              # C_m / pF
        V_thresh - 15, # V_thresh / mV
        0.0,              # p_V_reset
        0.5,              # Delta_T / mV
        5.0,              # tau_w / ms
        -5.0,            # a / nS
        -0.1,             # b / nA
    ], dtype=torch.float32)

    high = torch.tensor([
        E_L_est + 10.0,   # E_L / mV
        550.0,            # R_m / mOhm
        150.0,             # C_m / pF
        V_thresh + 3,  # V_thresh / mV
        1.0,              # p_V_reset
        7.0,             # Delta_T / mV
        500.0,            # tau_w
        15.0,             # a / nS
        0.15,              # b / nA
    ], dtype=torch.float32)

    priors = sbi_utils.BoxUniform(low=low, high=high)
    return priors


def run_simulations_9D(prior, I_inj_exp, tau_ref, num_simulations, savename):
    # runs simulations with given prior
    # input 
    # priors are the prior boundaries prepared for SBI
    # I_inj_exp is the experimentally injected current in pA
    # tau_ref is the absolute refractory period in ms
    # num_simulations is the number of simulations
    # savename is the save name of the simulated data
    
    # output 
    # theta (num_sim, 9) are the drawn priors for every simulation
    # sims are the simulated V_m and spike_times_post for each theta
    
    def is_sim_valid(V_m, spike_times_post):
        # check if simulation is valid
        #Criteria:
        # 1) all voltages finite
        # 2) |V_m| ≤ 300 mV
        # 3) all spike_times_post are finite
        # 4) at least 3 spikes (len(spike_times_post) > 2) --> prevents histograms to be calculated
        # 5) fewer than 75 spikes (len(spike_times_post) < 75)
    
        # input 
        # V_m is the simulated voltage trace in mV
        
        # output
        # V_m_valid is the simulated voltage trace fullfilling the criterium |V_m| < 300 mV
        
        V_m = np.asarray(V_m) # ensure ndarray for np.isfinite / abs
        V_m_valid = np.isfinite(V_m).all() and (np.max(np.abs(V_m)) <= 300.0)
        spike_times_post = np.asarray(spike_times_post)
        spike_times_post_valid = np.isfinite(spike_times_post).all() and (len(spike_times_post) > 2) and (len(spike_times_post) < 75)
        
        return V_m_valid and spike_times_post_valid
    
    # sample parameters
    theta_full = prior.sample((num_simulations,))
    
    theta_valid = []
    sims_valid = []
    dropped_num_sim = 0
    
    for i, param in enumerate(theta_full):
        
        E_L_mV, R_m_MOhm, C_m_pF, V_thresh_mV, p_V_reset, Delta_T_mV, tau_w_ms, a_nS, b_nA = [param[j].item() for j in range(9)] # unpack
        V_reset_mV = define_reasonable_V_reset(p_V_reset, E_L_mV, V_thresh_mV)

        # simulate
        time_sim, V_m_sim, spike_times_sim = AdExp_I_inj(I_inj_exp * pA, C_m_pF * pF, R_m_MOhm * Mohm, E_L_mV * mV, V_thresh_mV * mV, V_reset_mV * mV, Delta_T_mV * mV, tau_w_ms * ms, a_nS * nS, b_nA * nA, tau_ref * ms)
        
        if not is_sim_valid(V_m_sim, spike_times_sim):
            dropped_num_sim += 1
            continue  # skip saving this one
            
        sims_valid.append({
            "V_m": np.asarray(V_m_sim, dtype=np.float32),
            "spike_times": np.asarray(spike_times_sim, dtype=np.float32)})
        theta_valid.append(param)
        
    kept_num_sims = len(sims_valid)
    
    # avoid crash if everything got dropped
    if kept_num_sims == 0:
        print(f"[run_simulations_9D] WARNING: 0/{num_simulations} valid (0.0%). Not saving a file.")
        return torch.empty((0, 9), dtype=torch.float32), []
    
    theta = torch.stack(theta_valid, dim=0)
    sims = sims_valid
    
    
    # save
    os.makedirs("../Data_SBI_logs", exist_ok=True)
    path = f"../Data_SBI_logs/{savename}_n{num_simulations}.pkl"
    with open(path, "wb") as f:
        pkl.dump({"theta": theta, "sims": sims, "meta": {"requested": int(num_simulations), "kept": int(kept_num_sims), "dropped": int(dropped_num_sim), "validity": {"absV_max_mV": 300.0, "max_spikes": 75}}}, f)
    pct = 100.0 * kept_num_sims / float(num_simulations)
    print(f"[run_simulations_9D] Saved {kept_num_sims} valid ({pct:.1f}%) out of {num_simulations} to {path}")
    
    return theta, sims


def build_features_9D(V_m, spike_times_post, V_thresh, t_vec, dt_original=0.1, dt_target_V_m=2.0, dt_target_spike_train=125.0, pre_spike_time_window=5.0, std_ms=0.5, post_spike_time_window=20, tau_ref=0):
    # build features for stage 1 with V_m statistics (V_mean, V_median, V_std, n_spikes) + STA
    
    # input
    # V_m is the simulated or experimental voltage trace in mV
    # spike_times_post are the spike times in ms
    # V_thresh is the estimated threshold in mV used as spike cut off for STA alignment
    # t_vec is the time vector in ms
    # dt_original is the time step of the given voltage trace in ms
    # dt_target_V_m is the target time step for the voltage trace in ms
    # dt_target_spike_train is the target time step for the spike times in ms
    # pre_spike_time_window is the time window for STA in ms
    # std_ms is the standard deviation for Gaussian smoothing of the STAs in ms
    # post_spike_time_window is the time window for ASA in ms (default: 5.0)
    
    # output
    # feat is the feature array

    V_m_stats = V_m_statistics(V_m, spike_times_post)
    V_m_stats_tensor = np.asarray(V_m_stats, dtype=np.float32)
    
    STA, STA_segs = calculate_STA(V_m, spike_times_post, V_thresh, dt_original=dt_original, pre_spike_time_window=pre_spike_time_window, std_ms=std_ms)
    STA_standardized = STA - V_m_stats[0]
    STA_standardized_tensor = STA_standardized.astype(np.float32)
    #STA_tensor = STA.astype(np.float32)
    
    spike_times_stats = spike_times_statistics(spike_times_post)
    spike_times_stats_tensor = np.asarray(spike_times_stats, dtype=np.float32)
    ASA, _ = calculate_ASA(V_m, spike_times_post, V_thresh=V_thresh, post_spike_time_window=post_spike_time_window, dt_original=dt_original, std_ms=std_ms, tau_ref=tau_ref)
    ASA_standardized = ASA - V_m_stats[0]
    ASA_standardized_tensor = np.asarray(ASA_standardized, dtype=np.float32)
    #ASA_tensor = np.asarray(ASA, dtype=np.float32)
    _, t_vec_binarized, _, spike_train_binarized, spike_train_counts, spike_train_rates = process_voltage_trace_spike_times_post(t_vec, V_m, spike_times_post, V_thresh, dt_original=dt_original, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train)
    spike_train_rates_tensor = np.asarray(spike_train_rates, dtype=np.float32)
    
    
    #feat = np.concatenate([V_m_stats_tensor, spike_times_stats_tensor, spike_train_rates_tensor], axis=0)
    feat = np.concatenate([V_m_stats_tensor, STA_standardized_tensor, spike_times_stats_tensor, ASA_standardized_tensor, spike_train_rates_tensor], axis=0)
    #feat = np.concatenate([V_m_stats_tensor, STA_tensor, spike_times_stats_tensor, ASA_tensor, spike_train_rates_tensor], axis=0)
    return feat


def run_SBI_9D(V_m_exp, I_inj_exp, spike_times_post_exp, V_thresh, tau_ref, t_vec, dt_original, dt_target_V_m, dt_target_spike_train, pre_spike_time_window, post_spike_time_window, std_ms, T_window, prior, num_simulations, savename, time_limit=1200): 
    # performs SBI to estimate 9 AdExp parameters
    #  - loads the raw simulations (voltage trace + spike times) saved by run_simulations_9D
    #  - converts them to features using build_features_9D(...)
    #  - trains SNPE and samples with timeout + direct→MCMC fallback
    
    # input
    # V_m_exp is the experimental membrane voltage in mV
    # I_inj_exp is the experimentally injected current in pA
    # spike_times_post_exp is the experimentally measured spike times in ms
    # V_thresh is the estimated spiking threshold in mV (for STA alignment on experimental data)
    # tau_ref is the absolute refractory period in ms
    # t_vec is the time vector in ms
    # dt_original is the resolution of the voltage trace in ms
    # dt_target_V_m is the target time step for the voltage trace in ms
    # dt_target_spike_train is the target time step for the spike times in ms
    # pre_spike_time_window is the STA window in ms
    # post_spike_time_window is the time window for ASA in ms (default: 5.0)
    # std_ms is the standard deviation (ms) for Gaussian smoothing of the STAs & ASAs
    # T_window is the simulation time in ms (kept for symmetry; not directly used here)
    # prior is the BoxUniform prior over the 9D parameter vector
    # num_simulations is the number of simulations (default: 100)
    # savename is the cache filename stem used by run_simulations_9D
    # time_limit is the timeout for posterior sampling in seconds
    
    # output
    # maps are the maximum a posteriori estimated values 
    # means are the mean estimated values 
    # stds are the standard deviations of the estimated values
    # samples are samples of the posterior distribution
    
    # define time outs
    
    class _SBITimeout(Exception): ...
    def _timeout_handler(signum, frame): raise _SBITimeout
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
    except Exception:
        pass

    def _with_timeout(seconds, fn, *f_args, **f_kwargs):
        if hasattr(signal, "alarm"):
            prev = signal.alarm(0)
            try:
                signal.alarm(int(seconds))
                return fn(*f_args, **f_kwargs)
            finally:
                signal.alarm(0)
                if prev:
                    signal.alarm(prev)
        else:
            return fn(*f_args, **f_kwargs)
        
    # load raw sims
    path = f"../Data_SBI_logs/{savename}_n{num_simulations}.pkl"
    with open(path, "rb") as f:
        data = pkl.load(f)
    theta, sims = data["theta"], data["sims"]

    # build training features
    X = []
    for s in sims:
        V = s["V_m"]
        spikes = s["spike_times"]
        t_vec_sim = np.arange(len(V), dtype=np.float32) * float(dt_original)
        #t_vec = np.arange(len(V), dtype=np.float32) * float(dt_original)
        feat = build_features_9D(V_m=V, spike_times_post=spikes, V_thresh=V_thresh, t_vec=t_vec_sim, dt_original=dt_original, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train, pre_spike_time_window=pre_spike_time_window, std_ms=std_ms, post_spike_time_window=post_spike_time_window, tau_ref=tau_ref)
        X.append(torch.tensor(feat, dtype=torch.float32))
    x = torch.stack(X, dim=0)

    # train 
    # maybe leave out?
    theta = theta.detach().cpu().float() if hasattr(theta, "detach") else theta.float()
    x = x.detach().cpu().float() if hasattr(x, "detach") else x.float()
    
    
    inf = sbi_inference.SNPE(prior=prior)
    de = inf.append_simulations(theta, x).train()
    
    # z-scoring off
    #density_estimator = posterior_nn(model="nsf", z_score_x="none", z_score_theta="none")
    #inf = sbi_inference.SNPE(prior=prior, density_estimator=density_estimator)
    #de = inf.append_simulations(theta, x, exclude_invalid_x=True).train()
    
    post_direct = inf.build_posterior(de, sample_with="direct")
    post_mcmc   = inf.build_posterior(de, sample_with="mcmc", mcmc_method="slice_np", mcmc_parameters={"num_chains": 1, "thin": 10, "warmup_steps": 500})

    # experimental features
    feat_x_exp_np = build_features_9D(V_m=V_m_exp, spike_times_post=spike_times_post_exp, V_thresh=V_thresh, t_vec=t_vec, dt_original=dt_original, dt_target_V_m=dt_target_V_m, dt_target_spike_train=dt_target_spike_train, pre_spike_time_window=pre_spike_time_window, std_ms=std_ms, post_spike_time_window=post_spike_time_window, tau_ref=tau_ref)
    x_exp = torch.tensor(feat_x_exp_np, dtype=torch.float32)

    # sample with timeout (direct → MCMC)
    samples = None
    try:
        samples = _with_timeout(time_limit, post_direct.sample, (1000,), x=x_exp, show_progress_bars=False)
        print("[SBI 9D] direct sampling finished within time limit.")
        used = "direct"
    except _SBITimeout:
        print("[SBI 9D][TIMEOUT] direct too slow → switching to MCMC.")
        try:
            samples = _with_timeout(time_limit, post_mcmc.sample, (1000,), x=x_exp, show_progress_bars=False)
            print("[SBI 9D] MCMC finished within time limit.")
            used = "mcmc"
        except _SBITimeout:
            print("[SBI 9D][TIMEOUT] MCMC also timed out → returning NaNs.")
            means = np.full(9, np.nan, dtype=np.float32)
            stds  = np.full(9, np.nan, dtype=np.float32)
            maps  = [np.nan]*9
            return means, stds, maps, None

    # stats
    logp = (post_direct.log_prob(samples, x=x_exp) if used == "direct" else post_mcmc.log_prob(samples, x=x_exp))
    best_idx = torch.argmax(logp)
    maps  = samples[best_idx].detach().cpu().numpy().tolist()
    means = samples.mean(dim=0).detach().cpu().numpy()
    stds  = samples.std(dim=0).detach().cpu().numpy()
    
    # translate p_V_reset (index 4) to V_reset in mV
    p_V_reset_map  = maps[4]
    p_V_reset_mean = float(means[4])
    p_V_reset_std  = float(stds[4])

    V_reset_map, V_reset_mean, V_reset_std = translate_p_to_V_reset(p_V_reset_map, p_V_reset_mean, p_V_reset_std, maps[0], maps[3])
    
    # overwrite the p_V_reset entries with V_reset stats in mV
    maps[4]  = V_reset_map
    means[4] = V_reset_mean
    stds[4]  = V_reset_std
    
    # prepare samples with correct V_reset column (map p_V_reset -> V_reset in mV)
    p_V_reset_sample = samples[:, 4]
    V_reset_sample_mV = define_reasonable_V_reset(p_V_reset_sample, samples[:, 0], samples[:, 3])
    #V_reset_sample_mV = (1.0 - p_V_reset_sample) * (samples[:, 0] - 10.0) + p_V_reset_sample * (samples[:, 3] - 1.0)

    # replace p_V_reset with V_reset (mV)
    samples[:, 4] = V_reset_sample_mV


    print("MAP: E_L={:.2f} mV, R_m={:.2f} MOhm, C_m={:.2f} pF, V_thresh={:.2f} mV, V_reset={:.2f} mV, Delta_T={:.2f} mV, tau_w={:.2f} ms, a={:.2f} nS, b={:.3f} nA".format(*maps))

    return means, stds, maps, samples

##### other old functions ####

def OLD_update_efficiency_calculation_experimental_data_Zeldenrust(data_exp_folder,data_analyzed_folder):
    # calculate efficiency measures of all analyzed files and update analyzed data files
    
    # input
    # data_exp_folder is the path to the folder containing raw .mat files
    # data_analyzed_folder is the path to the folder containing analyzed .pkl files

    # load analyzed data 
    analyzed_dict = load_analyzed_data(data_analyzed_folder)

    for key, analyzed_data in analyzed_dict.items():

        #MI_FZ_chunked = analyzed_dict['MI_FZ_chunked']
        #firing_rate_FZ_chunked = analyzed_dict['firing_rate_FZ_chunked']

        analyzed_data['MICE_calculated'] = MI_calculated_bits / firing_rate_Hz
        analyzed_data['MICE_FZ'] = MI_FZ_bits / firing_rate_Hz
        analyzed_data['MI_calculated_per_energy'] = MI_calculated_bits / E_tot
        analyzed_data['MI_FZ_per_energy'] = MI_FZ_bits / E_tot
        analyzed_data['MICE_calculated_per_energy'] = MI_calculated_bits / firing_rate_Hz / E_tot
        analyzed_data['MICE_FZ_per_energy'] = MI_FZ_bits / firing_rate_Hz / E_tot

        analyzed_path = os.path.join(data_analyzed_folder, key + ".pkl")
        with open(analyzed_path, "wb") as f:
            pkl.dump(analyzed_data, f)

        print(f"Updated '{key}.pkl' with efficiency.")
        



def OLD_simulate_V_m_SBI_samples(I_inj_exp, SBI_means, SBI_stds, E_L_est, V_thresh, tau_ref, n_samples=100):
    # simulate V_m for mean SBI parameters and additional samples drawn stds

    # input
    # I_inj_exp is the experimental injected current in pA
    # SBI_means is a list of the mean SBI results ([R_m (Mohm), C_m (pF), V_reset (mV), Delta_T (mV), tau_w (ms), a (nS), b (nA)]
    # SBI_stds is a list of the standard deviation of the SBI results 
    # E_L_est is leak reversal potential in mV
    # V_thresh is the spiking threshold in mV
    # tau_ref is the refractory period in ms 
    # n_samples is the number of parameter samples to draw

    # output
    # t_ms is the time vector in ms
    # V_m_mean is the V_m trace for mean parameters in mV
    # V_m_samples are the n_samples V_m traces for samples in mV

    SBI_means = np.asarray(SBI_means, dtype=float)
    SBI_stds = np.asarray(SBI_stds, dtype=float)

    # central / mean parameters
    R_m_mean   = SBI_means[0]
    C_m_mean   = SBI_means[1]
    Delta_T_m  = SBI_means[2]
    V_reset_m  = SBI_means[3]
    tau_w_m    = SBI_means[4]
    a_m        = SBI_means[5]
    b_m        = SBI_means[6]

    t_vec_mean_sim, V_m_mean_sim, spike_times_post_mean_sim = AdExp_I_inj(I_inj_exp * pA, C_m_mean * pF, R_m_mean * Mohm, E_L_est * mV, V_thresh * mV, V_reset_m * mV, Delta_T_m * mV, tau_w_m * ms, a_m * nS, b_m * nA, tau_ref * ms)

    # convert to plain arrays
    T = len(V_m_mean_sim)

    # simulate for sampled parameter sets
    V_m_std_sim = np.empty((n_samples, T), dtype=float)
    spike_times_post_std_sim = []

    for k in range(n_samples):
        # draw one parameter vector from diagonal Gaussian
        theta = np.random.normal(loc=SBI_means, scale=SBI_stds)

        R_m   = theta[0]
        C_m   = theta[1]
        Delta_T = theta[2]
        V_reset = theta[3]
        tau_w = theta[4]
        a = theta[5]
        b = theta[6]

        t_vec_k, V_m_k, spikes_k = AdExp_I_inj(I_inj_exp * pA, C_m * pF, R_m * Mohm, E_L_est * mV, V_thresh * mV, V_reset * mV, Delta_T * mV, tau_w * ms, a * nS, b * nA, tau_ref * ms)

        V_m_std_sim[k, :] = V_m_k.astype(float)
        spike_times_post_std_sim.append(spikes_k)

    return t_vec_mean_sim, V_m_mean_sim, V_m_std_sim, spike_times_post_mean_sim, spike_times_post_std_sim

def OLD_plot_V_m_SBI_mean_std(t_ms, V_m_mean, V_m_samples, color="blue", ylims=None, figsize=(5, 4), ax=None, savename=None):
    # plot mean V_m trace with ±1 std from sampled parameter traces as a shaded band
    
    # input
    # t_ms is the time vector in ms 
    # V_m_mean is the membrane potential trace for mean parameters in mV
    # V_m_samples are n_samples standard deviation traces in mV
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

    # compute ±1 std envelope from samples
    if V_m_samples is not None and len(V_m_samples) > 0:
        V_m_std = np.std(V_m_samples, axis=0)
    else:
        V_m_std = np.zeros_like(V_m_mean)

    # main (mean) trace
    ax.plot(t_ms[:10000], V_m_mean[:10000], lw=1.0, alpha=1.0, color=color)
        
    # shaded area here: ±1 std around the mean trace
    ax.fill_between(t_ms[:10000], V_m_mean[:10000] - 2*V_m_std[:10000], V_m_mean[:10000] + 2*V_m_std[:10000], alpha=0.5, color='red', linewidth=1.0) 

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

    scale_time_ms = 200  # time scalebar length in ms
    scale_time_dt = scale_time_ms  # same, but in ms on x-axis
    scale_V_m_mV = 25   # voltage scalebar length in mV

    # time scalebar
    ax.plot([x0, x0 + scale_time_dt], [y0, y0], lw=4.0, color='black', clip_on=False, zorder=5)
    ax.text(x0 + scale_time_dt / 2.0, y0 - 0.04 * yr, f'{int(scale_time_ms)} ms', ha='center', va='top', clip_on=False)

    # voltage scalebar
    ax.plot([x0, x0], [y0, y0 + scale_V_m_mV], lw=4.0, color='black', clip_on=False, zorder=5)
    ax.text(x0 - 40.0, y0 + scale_V_m_mV / 2.0, f'{int(scale_V_m_mV)} mV', ha='right', va='center', rotation=90, clip_on=False)

    # save & plot standalone figures
    if standalone is True:
        if savename is not None:
            path = f"../Figures/{savename}.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved figure to {path}")
        plt.show()
        
        
def OLD_mutual_information_tuning_curve(spike_times_e, w_e, tuning_curve, N_e_noise):
    # calculate mutual information based on tuning curve (& weighted synaptic input)
    
    # input
    # spike_times_e are the presynaptic spike times in s
    # w_e is an array of synaptic weights
    # tuning_curve is a list of rates for respective orientations in Hz
    # N_e_noise is the number excitatory noise synapses
    
    # output
    # MI_tuning_curve is the mutual information between weighted signalling input spike trains and neuronal response in bits

    # get spike times during stimulus presentation of presynaptic spike train
    spike_times_signal_e_pre = [calculate_tuning_curve(spike_times_pre/second*1000) for spike_times_pre in spike_times_e.values()]
    # weight the presynaptic signalling spike trains during stimulus presentation with their respective synaptic weight
    weighted_spike_times_signal_e_pre = w_e[N_e_noise:, np.newaxis]*spike_times_signal_e_pre[N_e_noise:] # weight the time binned presynaptic spike trains with the corresponding weight
    # sum all weighted presynaptic signalling spike trains during stimulus presentation
    flattened_weighted_spike_times_signal_e_pre = np.sum(weighted_spike_times_signal_e_pre, axis=0) # flatten the pre-synaptic binned spike trains to a single array
    # calculate mutual information between weighted presynaptic signalling spike trains & postsynaptic response during stimulus presentation
    MI_tuning_curve = mutual_information(flattened_weighted_spike_times_signal_e_pre, tuning_curve)
    #print(MI_tuning_curve)
    #pf.plot_tuning_curves(flattened_weighted_spike_times_signal_e_pre, tuning_curve, 'stimulus', 'neuronal response', normalized=True, color_CTR = 'blue', color_FR = 'orange', savename=None)

    return MI_tuning_curve


        
        