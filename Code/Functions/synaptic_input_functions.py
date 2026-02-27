# synaptic input functions

# import packages for simulation and calculation
import numpy as np
from brian2 import *
# brian2 is unstable with python 3.12. In case of errors either use: 
#prefs.codegen.target = "numpy"  # or
import os
os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

############################# synaptic input functions Padamsey #############################
def synaptic_parameters(synaptic_scaling_mode='multiplicative'): 
    # return synaptic parameters
    
    # input
    # synaptic_scaling_mode decides if synaptic weight distribution is scaled multiplicatively or non-multiplicatively
    
    # output
    # N_syn is the total number of synaptic inputs
    # N_e is the number of excitatory synaptic inputs
    # N_e_signal_ratio is the percentage of excitatory coding neurons
    # N_i is the number of inhibitory synaptic inputs
    # rate_background_ratio is the background firing rate for grey screen in Hz
    # mu_rate_e is the mean of excitatory synaptic firing rate distribution in Hz
    # sigma_rate_e is the std of excitatory synaptic firing rate distribution in Hz
    # mu_rate_i is the mean of inhibitory synaptic firing rate distribution in Hz
    # sigma_rate_i is the std of inhibitory synaptic firing rate distribution in Hz
    # mu_weight_e is the mean of excitatory synaptic weight distribution
    # sigma_weight_e is the std of excitatory synaptic weight distribution
    # mu_weight_i is the mean of inhibitory synaptic weight distribution
    # sigma_weight_i is the std of inhibitory synaptic weight distribution
    
    # define synaptic input 
    N_syn = 5000 # total number of synapses # 5000 according to Goetz et al. 2021
    N_e = int(0.75*N_syn) # number of excitatory neurons # 0.75*N_syn according to Goetz et al. 2021
    N_e_signal_ratio = 0.5 # ratio of excitatory coding neurons # 0.8 or 3000 according to Goetz et al. 2021
    N_i = int(0.25*N_syn) # number of inhibitory neurons # 0.25*N_syn according to Goetz et al. 2021
    rate_background_ratio = 0.8 # background firing rate ratio
    
    mu_rate_e = 1.37 # mean of excitatory synaptic firing rate distribution in Hz # 1.37 according to Goetz et al. 2021
    sigma_rate_e = 0.23 # std of excitatory synaptic firing rate distribution in Hz # 0.23 according to Goetz et al. 2021
    mu_rate_i = 6 # Hz # mean of inhibitory synaptic firing rate distribution in Hz # 6 according to Goetz et al. 2021
    sigma_rate_i = 0 # std of inhibitory synaptic firing rate distribution in Hz # 0 according to Goetz et al. 2021
    
    mu_weight_e = 2.07 # mean of excitatory synaptic weight distribution # 2.07 according to Padamsey et al. 2022 ex vivo # -1.51 according to Goetz et al. 2021 
    sigma_weight_e = 0.64 # std of excitatory synaptic weight distribution # 0.64 according to Padamsey et al. 2022 ex vivo # 1.14 according to Goetz et al. 2021
    mu_weight_i = mu_weight_e # mean of inhibitory synaptic weight distribution
    sigma_weight_i = sigma_weight_e # std of inhibitory synaptic weight distribution
    
    if synaptic_scaling_mode == 'multiplicative': 
        return N_syn, N_e, N_e_signal_ratio, N_i, rate_background_ratio, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i
    
    if synaptic_scaling_mode == 'non-multiplicative': 
        
        mu_weight_e_CTR = mu_weight_e
        sigma_weight_e_CTR = sigma_weight_e
        mu_weight_i_CTR = mu_weight_i
        sigma_weight_i_CTR = sigma_weight_i # std of inhibitory synaptic weight distribution
        
        mu_weight_e_FR = 1.54  # mean of excitatory synaptic weight distribution # 1.54 according to Padamsey et al. 2022 ex vivo
        sigma_weight_e_FR = 0.49 # std of excitatory synaptic weight distribution # 0.49 according to Padamsey et al. 2022 ex vivo
        mu_weight_i_FR = mu_weight_e_FR # mean of inhibitory synaptic weight distribution
        sigma_weight_i_FR = sigma_weight_e_FR # std of inhibitory synaptic weight distribution
        
        return N_syn, N_e, N_e_signal_ratio, N_i, rate_background_ratio, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e_CTR, sigma_weight_e_CTR, mu_weight_i_CTR, sigma_weight_i_CTR, mu_weight_e_FR, sigma_weight_e_FR, mu_weight_i_FR, sigma_weight_i_FR


def homogeneous_synaptic_input(T, N_e_noise, N_i, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i):
    # create synaptic input weights and firing-rates for inhibitory and excitatory noise input with homogeneous Poisson process 
    
    # input
    # T is the duration of the simulation in ms
    # N_e_noise is the number excitatory noise synapses
    # N_i is the number of inhibitory synaptic inputs
    # mu_rate_e is the mean of excitatory synaptic firing rate distribution in Hz
    # sigma_rate_e is the std of excitatory synaptic firing rate distribution in Hz
    # mu_rate_i is the mean of inhibitory synaptic firing rate distribution in Hz
    # sigma_rate_i is the std of inhibitory synaptic firing rate distribution in Hz
    # mu_weight_e is the mean of excitatory synaptic weight distribution
    # sigma_weight_e is the std of excitatory synaptic weight distribution
    # mu_weight_i is the mean of inhibitory synaptic weight distribution
    # sigma_weight_i is the std of inhibitory synaptic weight distribution
    
    # output 
    # w_e_0 is an array of length N_e with normalized excitatory synaptic weights
    # w_i_0 is an array of length N_i with normalized inhibitory synaptic weights
    # spike_times_e is a dictionary where keys are neuron indices and values are arrays of spike times for excitatory neurons
    # spike_times_i is a dictionary where keys are neuron indices and values are arrays of spike times for inhibitory neurons
    # r_e is an array of length N_e with firing rates of excitatory neurons
    # r_i is an array of length N_i with firing rates of inhibitory neurons 
    
    # distribution of synaptic weights
    w_e_raw = np.random.lognormal(mu_weight_e, sigma_weight_e, size = N_e_noise)
    w_e_0 = w_e_raw 
    
    w_i_raw = np.random.lognormal(mu_weight_i, sigma_weight_i, size = N_i)
    w_i_0 = w_i_raw 
    
    # distribution of input firing rates and homogeneous Poisson spike trains  
    r_e = np.random.normal(mu_rate_e, sigma_rate_e, size = N_e_noise)*Hz 
    P_e = PoissonGroup(N_e_noise, rates=r_e)
    spike_mon_e = SpikeMonitor(P_e)
    
    r_i = np.random.normal(mu_rate_i, sigma_rate_i, size = N_i)*Hz
    P_i = PoissonGroup(N_i, rates=r_i)
    spike_mon_i = SpikeMonitor(P_i)
        
    net = Network(P_e, P_i, spike_mon_e, spike_mon_i)
    net.run(T * ms)
    
    spike_times_e = spike_mon_e.spike_trains()
    spike_times_i = spike_mon_i.spike_trains()
    
    return w_e_0, w_i_0, spike_times_e, spike_times_i, r_e, r_i

def orientation_variation_excitatory_input(N_e_signal, mu_rate_e, sigma_rate_e, rate_background_ratio, mu_weight_e, sigma_weight_e):
    # create synaptic input weights and firing-rates for a variation in presented grating orientation as input for an inhomogeneous Poisson process
    # continous variation of presented stimulus represented by different firing-rate distributions

    # in experiment they use 1.5s of stimulus presentation intermitted by 1.5s of background presentation
    # in experiment they measured 33 x 2 = 66 grating presentations per CTR and FR group
    # 3s background, 1.5 s 0 stimulus, 1.5s background, 1.5 s 30 stimulus, 1.5s background, 1.5 s 60 stimulus, 1.5s background, 1.5s 90 stimulus, 1.5s background, 1.5 s 120 stimulus, 1.5s background, 1.5 s 150 stimulus, 3s background
    # we use a simplification with [0,45,90,135,180] degree orientations with 1.5s background intermittent background presentation
    
    # input
    # N_e_signal is the number excitatory signaling synaptic neurons
    # mu_rate_e is the mean of excitatory synaptic firing rate distribution in Hz
    # sigma_rate_e is the std of excitatory synaptic firing rate distribution in Hz
    # rate_background_ratio ratio of mean excitatory rate to be used as background stimulation
    # mu_weight_e is the mean of excitatory synaptic weight distribution
    # sigma_weight_e is the std of excitatory synaptic weight distribution
    
    # output 
    # w_e_0 is an array of length N_e with normalized and sorted excitatory synaptic weights
    # r_e is a list of an arrays with the varying spiking frequencies of excitatory synaptic inputs in Hz
    # t_stim is an array with the values in time of presented grating orientation
    # T is the duration of the simulation in ms
    
    # distribution of synaptic weights  
    
    w_e_raw = np.random.lognormal(mu_weight_e, sigma_weight_e, size = N_e_signal)
    w_e_raw_sorted = np.sort(w_e_raw)
    w_e_0 = w_e_raw_sorted #/ sum(w_e_raw_sorted)  # normalize excitatory input
    
    # distribution of input firing rates - inhomogeneous Poisson process 

    grating_orientations = [0,30,60,90,120,150,180] #[0,45,90,135,180] # 
    T_presentation = 1500 # in ms
    T_background = 1500 # in ms

    # initialize firing rates
    r_e_background = np.ones((N_e_signal, T_background)) * mu_rate_e * rate_background_ratio 
    t_stim_background = np.array([None] * T_background, dtype=object) # track stimulus over time

    # initialize excitatory firing rates
    r_e_total = [np.concatenate((arr1, arr2)) for arr1, arr2 in zip(r_e_background, r_e_background)]
    t_stim_total = np.concatenate([t_stim_background, t_stim_background])
    
    # iterate through orientations
    for grating_orientation in grating_orientations:
        distance_to_90 = np.abs(grating_orientation - 90) % 180  # distance to 90, symmetrical around 90
        preference_ratio = 1 - (distance_to_90 / 90)  # linear decrease from 1 at 90 degrees to 0 at 0 degree and 180 degree
        print('Grating orientation: ' + str(grating_orientation) + ' degree , preference ratio: ' + str(round(preference_ratio,2)))
        
        # calculate number of synapses from preferred or nonpreferred distribution
        N_e_signal_synapses_from_preferred = int(N_e_signal * preference_ratio)
        N_e_signal_synapses_from_nonpreferred = N_e_signal - N_e_signal_synapses_from_preferred 

        # draw n and n-1 synaptic firing rates from (non-)preferred orientation distributions
        exc_synaptic_rates_preferred = np.random.normal(mu_rate_e, sigma_rate_e, size = N_e_signal_synapses_from_preferred) 
        exc_synaptic_rates_sorted_preferred = np.sort(exc_synaptic_rates_preferred)
        exc_synaptic_rates_nonpreferred = np.random.normal(mu_rate_e, sigma_rate_e, size = N_e_signal_synapses_from_nonpreferred) 
        exc_synaptic_rates_sorted_nonpreferred = np.sort(exc_synaptic_rates_nonpreferred)[::-1] # non-preferred orientation is reversed the preferred orientation
        exc_synaptic_rates_sorted = np.concatenate((exc_synaptic_rates_sorted_preferred, exc_synaptic_rates_sorted_nonpreferred))
        
        # add oriented grating stimulus
        r_e_grating_orientation = exc_synaptic_rates_sorted[:, np.newaxis] * np.ones((N_e_signal,T_presentation))
        t_stim_grating_orientation = np.ones(T_presentation) * preference_ratio # track stimulus over time
        r_e_total = [np.concatenate((arr1, arr2)) for arr1, arr2 in zip(r_e_total, r_e_grating_orientation)]
        t_stim_total = np.concatenate([t_stim_total, t_stim_grating_orientation])

        # add background stimulation
        r_e_total = [np.concatenate((arr1, arr2)) for arr1, arr2 in zip(r_e_total, r_e_background)]
        t_stim_total = np.concatenate([t_stim_total, t_stim_background])

    # add background in the end of simulation time --> omit for faster simulations
    #r_e_total = [np.concatenate((arr1, arr2)) for arr1, arr2 in zip(r_e_total, r_e_background)]
    #t_stim_total = np.concatenate([t_stim_total, t_stim_background])
    
    r_e = np.maximum(r_e_total,0) # restrict rates to positive values
    
    T = int(len(t_stim_total)) # calculate length of simulation
    t_stim = t_stim_total
    
    # creates a list of NumPy arrays of inhomogeneous input spike trains of spike times
    spike_times_e = []

    # generate spike trains for excitatory neurons
    for neuron_rate in r_e:
        p_spike = neuron_rate / 1000 # divided by 1000 to translate Hz into ms
        spikes = np.random.rand(T) < p_spike # generate spikes based on probabilities
        spike_times = np.nonzero(spikes)[0] # convert spikes to spike times
        spike_times_e.append(spike_times)
    
    return w_e_0, spike_times_e, r_e, t_stim, T

def orientation_variation_input(N_e, N_i, N_e_signal_ratio, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, rate_background_ratio, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i):
    # input
    # N_e is the number excitatory synaptic connections
    # N_i is the number inhibitory synaptic connections
    # N_e_signal_ratio portion of signalling synapses
    # mu_rate_e is the mean of excitatory synaptic firing rate distribution in Hz
    # sigma_rate_e is the std of excitatory synaptic firing rate distribution in Hz
    # mu_rate_i is the mean of inhibitory synaptic firing rate distribution in Hz
    # sigma_rate_i is the std of inhibitory synaptic firing rate distribution in Hz
    # rate_background_ratio ratio of mean excitatory rate to be used as background stimulation
    # mu_weight_e is the mean of excitatory synaptic weight distribution
    # sigma_weight_e is the std of excitatory synaptic weight distribution
    # mu_weight_i is the mean of inhibitory synaptic weight distribution
    # sigma_weight_i is the std of inhibitory synaptic weight distribution
    
    # output 
    # w_e_0 is an array of length N_e with normalized excitatory synaptic weights
    # w_i_0 is an array of length N_i with normalized inhibitory synaptic weights
    # spike_times_e is a list of arrays of spike times for excitatory neurons
    # spike_times_i is a list of arrays of spike times for inhibitory neurons
    # r_e is a list of N_e arrays of length T with firing rates of excitatory neurons
    # r_i is an array of length N_i with firing rates of inhibitory neurons 
    # t_stim is an array with the values in time of presented grating orientation
    # T is the duration of the simulation in ms
    
    N_e_signal = int(N_e_signal_ratio*N_e) # number of exc. coding neurons, 3000 according to Goetz et al. 2021
    N_e_noise = int(N_e - N_e_signal) # number of exc. noise neurons, 3750 according to Goetz et al. 2021

    # create excitatory oriented grating input for N_e_signal signalling synapses
    w_e_0_signal, spike_times_e_signal, r_e_signal, t_stim, T = orientation_variation_excitatory_input(N_e_signal, mu_rate_e, sigma_rate_e, rate_background_ratio, mu_weight_e, sigma_weight_e)
    
    # create excitatory noise input for N_e_noise noise synapses and N_i inhibitory synapses
    w_e_0_noise, w_i_0, spike_times_e_noise, spike_times_i, r_e_noise, r_i = homogeneous_synaptic_input(T, N_e_noise, N_i, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i)
    w_i_0 = w_i_0 / sum(w_i_0) # normalize excitatory input to scale input later
    
    # add excitatory signalling and noise synaptic weights
    w_e_0 = np.concatenate((w_e_0_noise, w_e_0_signal))
    w_e_0 = w_e_0 / sum(w_e_0) # normalize excitatory input to scale input later

    # add excitatory signalling and noise spike trains
    spike_times_e = spike_times_e_noise.copy()
    for i, spike_times in enumerate(spike_times_e_signal):
        spike_times_e[N_e_noise+i] = spike_times * 0.001 * second

    # add excitatory signalling and noise rates
    r_e_noise_mat = np.repeat(r_e_noise[:, np.newaxis]/Hz, T, axis=1)
    r_e = np.concatenate((r_e_noise_mat, r_e_signal))

    return w_e_0, w_i_0, spike_times_e, spike_times_i, r_e, r_i, t_stim, T, N_e_signal, N_e_noise



############################# synaptic input functions Zeldenrust #############################

def synaptic_parameters_switching(neuron_type_mode='excitatory'): 
    # return synaptic parameters
    
    # input
    # neuron_type_mode is a string ('excitatory' or 'inhibitory') determining the switching parameters
    
    # output
    # N_syn is the total number of synaptic inputs
    # N_e is the number of excitatory synaptic inputs
    # N_e_signal_ratio is the fraction of excitatory coding neurons
    # N_i is the number of inhibitory synaptic inputs
    # r_on is the switching rate of the hidden state in Hz from OFF to ON
    # r_off is the switching rate of the hidden state in Hz from ON to OFF
    # mu_rate_e is the mean of excitatory synaptic firing rate distribution in Hz
    # sigma_rate_e is the std of excitatory synaptic firing rate distribution in Hz
    # mu_rate_i is the mean of inhibitory synaptic firing rate distribution in Hz
    # sigma_rate_i is the std of inhibitory synaptic firing rate distribution in Hz
    # mu_weight_e is the mean of excitatory synaptic weight distribution
    # sigma_weight_e is the std of excitatory synaptic weight distribution
    # mu_weight_i is the mean of inhibitory synaptic weight distribution
    # sigma_weight_i is the std of inhibitory synaptic weight distribution
    
    # define synaptic input 
    N_syn = 1333 # total number of synapses # 5000 according to Goetz et al. 2021
    N_e = int(0.75*N_syn) # number of excitatory neurons # 1000 according to Zeldenrust et al. 2024 # 0.75*N_syn according to Goetz et al. 2021
    N_e_signal_ratio = 0.5 # ratio of excitatory coding neurons # 0.8 or 3000 according to Goetz et al. 2021
    N_i = int(0.25*N_syn) # number of inhibitory neurons # 0.25*N_syn according to Goetz et al. 2021
    
    # differentiate between excitatory & inhibitory cells
    if neuron_type_mode == 'excitatory': 
        r_on = 1.3 # Hz # switching rate of the hidden state in Hz from OFF to ON according to Zeldenrust et al. 2017
        r_off = 2.7  # Hz # switching rate of the hidden state in Hz from ON to OFF (2*r_on) according to Zeldenrust et al. 2017
    if neuron_type_mode == 'inhibitory': 
        r_on = 6.7 # Hz # switching rate of the hidden state in Hz from OFF to ON according to Zeldenrust et al. 2017 
        r_off = 13.3 # Hz # switching rate of the hidden state in Hz from ON to OFF according to Zeldenrust et al. 2017
    
    mu_rate_e = 1.37 # mean of excitatory synaptic firing rate distribution in Hz # 1.37 according to Goetz et al. 2021 similar in Zeldenrust et al. 2017
    sigma_rate_e = 0.23 # std of excitatory synaptic firing rate distribution in Hz # 0.23 according to Goetz et al. 2021 similar in Zeldenrust et al. 2017
    mu_rate_i = 6 # Hz # mean of inhibitory synaptic firing rate distribution in Hz # 6 according to Goetz et al. 2021
    sigma_rate_i = 0 # std of inhibitory synaptic firing rate distribution in Hz # 0 according to Goetz et al. 2021
    
    mu_weight_e = 2.07 # mean of excitatory synaptic weight distribution # 2.07 according to Padamsey et al. 2022 ex vivo # -1.51 according to Goetz et al. 2021 
    sigma_weight_e = 0.64 # std of excitatory synaptic weight distribution # 0.64 according to Padamsey et al. 2022 ex vivo # 1.14 according to Goetz et al. 2021
    mu_weight_i = mu_weight_e # mean of inhibitory synaptic weight distribution
    sigma_weight_i = sigma_weight_e # std of inhibitory synaptic weight distribution
        
    return N_syn, N_e, N_e_signal_ratio, N_i, r_on, r_off, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i

def switching_excitatory_input(N_e_signal, mu_rate_e, sigma_rate_e, mu_weight_e, sigma_weight_e, r_on, r_off, T):
    # create synaptic input weights and spike trains for excitatory neurons modulated by a switching hidden state

    # input
    # N_e_signal is the number of excitatory signaling synaptic neurons
    # mu_rate_e is the mean of excitatory synaptic firing rate distribution in Hz
    # sigma_rate_e is the std of excitatory synaptic firing rate distribution in Hz
    # mu_weight_e is the mean of excitatory synaptic weight distribution
    # sigma_weight_e is the std of excitatory synaptic weight distribution
    # r_on is the switching rate to ON state in Hz
    # r_off is the switching rate to OFF state in Hz
    # T is the duration of the simulation in ms

    # output 
    # w_e_0 is an array of length N_e_signal with sorted excitatory synaptic weights
    # spike_times_e is a list of arrays with spike times in ms
    # r_e is a list of arrays with firing rates in Hz
    # t_stim is an array with hidden state (ON/OFF) over time in ms

    t_stim = np.zeros(T, dtype=int) # hidden state 
    state = 0
    for t in range(1, T):
        if state == 0 and np.random.rand() < (r_on / 1000):
            state = 1
        elif state == 1 and np.random.rand() < (r_off / 1000):
            state = 0
        t_stim[t] = state

    # draw firing rates once
    r_e_raw = np.random.normal(mu_rate_e, sigma_rate_e, size=N_e_signal)
    r_e_sorted = np.sort(r_e_raw)
    
    # draw weights once
    w_e_raw = np.random.lognormal(mu_weight_e, sigma_weight_e, size=N_e_signal)
    w_e_sorted = np.sort(w_e_raw)
    w_e_0 = w_e_sorted
    
    # ON mapping -> same order
    r_e_on = r_e_sorted
    
    # OFF mapping -> reversed order
    r_e_off = r_e_sorted[::-1]
    
    # for each synapse generate ON/OFF switching rates based on hidden state
    r_e = []
    for i in range(N_e_signal):
        rates = np.where(t_stim == 1, r_e_on[i], r_e_off[i])
        r_e.append(rates)

    # generate spikes
    spike_times_e = []
    for neuron_rate in r_e:
        p_spike = neuron_rate / 1000
        spikes = np.random.rand(T) < p_spike
        spike_times = np.nonzero(spikes)[0]
        spike_times_e.append(spike_times)

    return w_e_0, spike_times_e, r_e, t_stim


def switching_input(N_e, N_i, N_e_signal_ratio, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i, r_on, r_off, T):
    # create synaptic input weights and spike trains for excitatory and inhibitory neurons modulated by a switching hidden state

    # input
    # N_e is the number of excitatory synaptic connections
    # N_i is the number of inhibitory synaptic connections
    # N_e_signal_ratio is the portion of signaling synapses
    # mu_rate_e is the mean of excitatory synaptic firing rate distribution in Hz
    # sigma_rate_e is the std of excitatory synaptic firing rate distribution in Hz
    # mu_rate_i is the mean of inhibitory synaptic firing rate distribution in Hz
    # sigma_rate_i is the std of inhibitory synaptic firing rate distribution in Hz
    # mu_weight_e is the mean excitatory synaptic weight
    # sigma_weight_e is the std excitatory synaptic weight
    # mu_weight_i is the mean inhibitory synaptic weight
    # sigma_weight_i is the std inhibitory synaptic weight
    # r_on is the switching rate to ON state in Hz
    # r_off is the switching rate to OFF state in Hz
    # T is the duration of the simulation in ms

    # output
    # w_e_0 is an array of normalized excitatory synaptic weights
    # w_i_0 is an array of normalized inhibitory synaptic weights
    # spike_times_e is a list of arrays with spike times for excitatory neurons in ms
    # spike_times_i is a list of arrays with spike times for inhibitory neurons in ms
    # r_e is a list of arrays of excitatory firing rates in Hz
    # r_i is an array of inhibitory firing rates in Hz
    # t_stim is an array with hidden state (ON/OFF) over time in ms
    # N_e_signal is the number of signaling excitatory synapses
    # N_e_noise is the number of noisy excitatory synapses

    N_e_signal = int(N_e_signal_ratio * N_e)
    N_e_noise = N_e - N_e_signal

    # create excitatory signalling weights & rates
    w_e_0_signal, spike_times_e_signal, r_e_signal, t_stim = switching_excitatory_input(N_e_signal, mu_rate_e, sigma_rate_e, mu_weight_e, sigma_weight_e, r_on, r_off, T)

    # create excitatory noise & inhibitory weights & rates
    w_e_0_noise, w_i_0, spike_times_e_noise, spike_times_i, r_e_noise, r_i = homogeneous_synaptic_input(T, N_e_noise, N_i, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i)
    w_i_0 = w_i_0 / sum(w_i_0)
    
    # add excitatory signalling and noise synaptic weights
    w_e_0 = np.concatenate((w_e_0_noise, w_e_0_signal))
    w_e_0 = w_e_0 / sum(w_e_0)

    # add excitatory signalling and noise spike trains
    spike_times_e = spike_times_e_noise.copy()
    for i, spike_times in enumerate(spike_times_e_signal):
        spike_times_e[N_e_noise + i] = spike_times * 0.001 * second

    # add excitatory signalling and noise rates
    r_e_noise_mat = np.repeat(r_e_noise[:, np.newaxis] / Hz, T, axis=1)
    r_e = np.concatenate((r_e_noise_mat, r_e_signal))

    return w_e_0, w_i_0, spike_times_e, spike_times_i, r_e, r_i, t_stim, N_e_signal, N_e_noise

