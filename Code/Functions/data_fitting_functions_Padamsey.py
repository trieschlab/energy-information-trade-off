# data fitting functions 

# import predefined functions
import Functions.synaptic_input_functions as sif
import Functions.simulation_functions as sf

# import packages for simulation and calculation
import numpy as np


################################ data fitting functions ################################

def single_r_post_output_run(w_scale_iteration, T, N_e, N_i, spike_times_e, spike_times_i, E_e, E_i, w_e_0, w_i_0, tau_e, tau_i, C_m, g_L_exp, E_L_exp, V_thresh, V_reset_exp, rate_window, Delta_T_ad, tau_w_ad, a_ad, b_ad, membrane_noise, model_mode):
    # run a single simulation & return r_post of iteration
    
    # input
    # w_scale_iteration is the w_scale parameter to try in iterative process
    # T is the duration of the simulation in ms
    # N_e is the number of excitatory synaptic inputs
    # N_i is the number of inhibitory synaptic inputs
    # spike_times_e is an array of excitatory input spike trains
    # spike_times_i is an array of inhibitory input spike trains
    # E_e is the reversal potential for excitatory inputs in mV
    # E_i is the reversal potential for inhibitory inputs in mV
    # w_e_0 are the weights of exc. input
    # w_i_0 are the weights of inh. input
    # tau_e is the postsynaptic potential (PSP) time constant for exc. input in ms
    # tau_i is the postsynaptic potential (PSP) time constant for inh. input in ms
    # C_m is the membrane capacitance in pF
    # g_L_exp is the experimentally measured leak conductance in nS
    # E_L_exp is the  experimentally measured resting potential (leak reversal potential) in mV
    # V_thresh is the spike generation threshold in mV
    # V_reset_exp is the  experimentally measured reset potential in mV
    # rate_window is the integration time for the online firing-rate estimation in ms
    # Delta_T_ad is the AdExp slope factor
    # tau_w_ad is the AdExp adaptation time constant
    # a_ad is the AdExp subthreshold adaptation
    # b_ad is the AdExp spike-triggered adaptation
    # membrane_noise is the std of the membrane noise in mV/s
    # model_mode is a list of strings or string & decides which model is used for simulation
    
    # output
    # r_post_simulation_LIF, r_post_simulation_AdExp are the simulated firing rates
        
    # initialize r_posts to save results
    r_post_simulation_LIF = None
    r_post_simulation_AdExp = None
    
    w_e = w_e_0 * 5 * w_scale_iteration * 4 # low E/I balance: 4/1
    w_i = w_i_0 * 5 * w_scale_iteration * 1 # low E/I balance: 4/1

    # translate parameter to Brian-readable values
    T_b, N_e, N_i, spike_times_e, spike_times_i, E_e_b, E_i_b, w_e_b, w_i_b, tau_e_b, tau_i_b, C_m_b, g_L_exp_b, E_L_exp_b, V_thresh_b, V_reset_exp_b, rate_window_b, Delta_T_ad_b, tau_w_ad_b, a_ad_b, b_ad_b, membrane_noise_b = sf.parameters_brian(T, N_e, N_i, spike_times_e, spike_times_i, E_e, E_i, w_e, w_i, tau_e, tau_i, C_m, g_L_exp, E_L_exp, V_thresh, V_reset_exp, rate_window, Delta_T_ad, tau_w_ad, a_ad, b_ad, membrane_noise)
    
    # simulate LIF if given    
    if 'LIF' in model_mode:
        
        # scale synaptic input
        w_e_LIF_b = w_e_b
        w_i_LIF_b = w_i_b

        # simulate LIF
        time_LIF, V_m_LIF, spike_times_post_LIF, rate_estimate_post_LIF, I_syn_e_LIF, I_syn_i_LIF, g_e_LIF, g_i_LIF = sf.LIF(T_b, N_e, N_i, spike_times_e, spike_times_i, E_e_b, E_i_b, w_e_LIF_b, w_i_LIF_b, tau_e_b, tau_i_b, C_m_b, g_L_exp_b, E_L_exp_b, V_thresh_b, V_reset_exp_b, rate_window_b, membrane_noise_b)
        
        # estimate r_post from LIF simualtion
        r_post_simulation_LIF = len(spike_times_post_LIF) / T * 1000 # is the postsynaptic firing rate in Hz
        
    # simulate AdExp if given  
    if 'AdExp' in model_mode:               

        # scale synaptic input
        weight_factor_LIF_to_AdExp = 1.5
        w_e_AdExp_b = w_e_b * weight_factor_LIF_to_AdExp
        w_i_AdExp_b = w_i_b * weight_factor_LIF_to_AdExp
        
        # simulate AdExp
        time_AdExp, V_m_AdExp, spike_times_post_AdExp, rate_estimate_post_AdExp, I_syn_e_AdExp, I_syn_i_AdExp, g_e_AdExp, g_i_AdExp = sf.AdExp(T_b, N_e, N_i, spike_times_e, spike_times_i, E_e_b, E_i_b, w_e_AdExp_b, w_i_AdExp_b, tau_e_b, tau_i_b, C_m_b, g_L_exp_b, E_L_exp_b, V_thresh_b, V_reset_exp_b, Delta_T_ad_b, tau_w_ad_b, a_ad_b, b_ad_b, rate_window_b, membrane_noise_b)
        
        # estimate r_post from AdExp simualtion
        r_post_simulation_AdExp = len(spike_times_post_AdExp) / T * 1000 # is the postsynaptic firing rate in Hz

    return r_post_simulation_LIF, r_post_simulation_AdExp

def single_w_scale_estimation_run(r_post_exp, w_scale_init, T, N_e, N_i, spike_times_e, spike_times_i, E_e, E_i, w_e_0, w_i_0, tau_e, tau_i, C_m, g_L_exp, E_L_exp, V_thresh, V_reset_exp, rate_window, Delta_T_ad, tau_w_ad, a_ad, b_ad, membrane_noise, model_mode, tolerance=0.0099, max_iterations=15):
    # estimate w_scale_simulation for r_post_exp values for single experimental value combination

    # input
    # r_post_exp is the experimentally measured firing rate in Hz
    # w_scale_init is the w_scale to start iteration with
    # T is the duration of the simulation in ms
    # N_e is the number of excitatory synaptic inputs
    # N_i is the number of inhibitory synaptic inputs
    # spike_times_e is an array of excitatory input spike trains
    # spike_times_i is an array of inhibitory input spike trains
    # E_e is the reversal potential for excitatory inputs in mV
    # E_i is the reversal potential for inhibitory inputs in mV
    # w_e_0 are the weights of exc. input
    # w_i_0 are the weights of inh. input
    # tau_e is the postsynaptic potential (PSP) time constant for exc. input in ms
    # tau_i is the postsynaptic potential (PSP) time constant for inh. input in ms
    # C_m is the membrane capacitance in pF
    # g_L_exp is the experimentally measured leak conductance in nS
    # E_L_exp is the  experimentally measured resting potential (leak reversal potential) in mV
    # V_thresh is the spike generation threshold in mV
    # V_reset_exp is the  experimentally measured reset potential in mV
    # rate_window is the integration time for the online firing-rate estimation in ms
    # Delta_T_ad is the AdExp slope factor
    # tau_w_ad is the AdExp adaptation time constant
    # a_ad is the AdExp subthreshold adaptation
    # b_ad is the AdExp spike-triggered adaptation
    # membrane_noise is the std of the membrane noise in mV/s
    # model_mode is a list of strings or string & decides which model is used for simulation
    # membrane_noise_mode decides if membrane noise is added or not
    # tolerance is the acceptable difference between experimental target and simulated r_post
    # max_iterations is the maximum number of iterations
    
    # output
    # w_scale_iteration is the weight value resulting from the iteration 
    # r_post_simulation_LIF, r_post_simulation_AdExp are the simulated firing rates corresponding to w_scale_iteration 

     
    # set the initial step size for adjusting the w_scale
    w_scale_iteration = w_scale_init 
    step_size = w_scale_iteration / 5
    iteration = 0

    while iteration < max_iterations:
        # simulate model with the current w_scale_iteration
        r_post_simulation_LIF, r_post_simulation_AdExp = single_r_post_output_run(w_scale_iteration, T, N_e, N_i, spike_times_e, spike_times_i, E_e, E_i, w_e_0, w_i_0, tau_e, tau_i, C_m, g_L_exp, E_L_exp, V_thresh, V_reset_exp, rate_window, Delta_T_ad, tau_w_ad, a_ad, b_ad, membrane_noise, model_mode)
        if model_mode == 'LIF':
            print(f"Iteration {iteration+1} with r_post_simulation = {r_post_simulation_LIF} and r_post_target = {r_post_exp} and w_scale = {w_scale_iteration}")
            
            # check if the simulated rate is within the tolerance
            if abs(r_post_simulation_LIF - r_post_exp) < tolerance:
                print(f"Converged to w_scale = {w_scale_iteration} after {iteration} iterations with r_post_simulation = {r_post_simulation_LIF} and r_post_target = {r_post_exp}.")
                return w_scale_iteration, r_post_simulation_LIF

            # adjust step size based on how far the simulation result is from the experimental target value
            error = r_post_simulation_LIF - r_post_exp
            step_size = max(step_size / 2, 0.00001)  # gradually reduce step size but avoid too small steps
            
            # adjust w_scale_iteration based on the simulation result
            if r_post_simulation_LIF < r_post_exp:
                w_scale_iteration += step_size  # increase w_scale if the simulated rate is too low
            else:
                w_scale_iteration -= step_size  # decrease w_scale if the simulated rate is too high
            
            iteration += 1
        
        if model_mode == 'AdExp':
            print(f"Iteration {iteration+1} with r_post_simulation = {r_post_simulation_AdExp} and r_post_target = {r_post_exp} and w_scale = {w_scale_iteration}")
            
            # check if the simulated rate is within the tolerance
            if abs(r_post_simulation_AdExp - r_post_exp) < tolerance:
                print(f"Converged to w_scale = {w_scale_iteration} after {iteration} iterations with r_post_simulation = {r_post_simulation_AdExp} and r_post_target = {r_post_exp}.")
                return w_scale_iteration, r_post_simulation_AdExp

            # adjust step size based on how far the simulation result is from the experimental target value
            error = r_post_simulation_AdExp - r_post_exp
            step_size = max(step_size / 2, 0.001)  # gradually reduce step size but avoid too small steps
            
            # adjust w_scale_iteration based on the simulation result
            if r_post_simulation_AdExp < r_post_exp:
                w_scale_iteration += step_size  # increase w_scale if the simulated rate is too low
            else:
                w_scale_iteration -= step_size  # decrease w_scale if the simulated rate is too high
            
            iteration += 1

    # if the maximum number of iterations is reached return the best estimate
    print(f"Maximum number of iterations reached, estimated w_scale so far = {w_scale_iteration}")
    if model_mode == 'LIF':
        return w_scale_iteration, r_post_simulation_LIF
    if model_mode == 'AdExp':
        return w_scale_iteration, r_post_simulation_AdExp

def full_w_scale_estimation_run(model_mode, R_m, E_L, r_post, T):
    # estimate w_scale from r_post values for multiple experimental values
    
    # input
    # model_mode is a list of strings or string & decides which model is used for simulation
    # R_m is a tuple of lists of (experimentally measured) membrane resistances equal to leak resistances in MOhm
    # E_L is a tuple of lists of (experimentally measured) resting potentials (leak reversal potentials) in mV
    # r_post is tuple of lists of (experimentally measured) firing rates in Hz
    # T is the total simulation time in ms
    # membrane_noise_mode decides if membrane noise is added or not

    # ouput 
    # w_scale_CTR, w_scale_FR are lists of the estimated weight scaling parameter for CTR & FR
    # r_post_simulation_CTR, r_post_simulation_FR  are lists of the simulated firing rates for CTR & FR
    
    # load parameter values
    # define homogeneous synaptic input
    N_syn, N_e, N_e_signal_ratio, N_i, rate_background_ratio, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i = sf.synaptic_parameters(synaptic_scaling_mode='multiplicative')
    w_e_0, w_i_0, spike_times_e, spike_times_i, r_e, r_i = sif.homogeneous_synaptic_input(T, N_e, N_i, mu_rate_e, sigma_rate_e, mu_rate_i, sigma_rate_i, mu_weight_e, sigma_weight_e, mu_weight_i, sigma_weight_i)
    
    # load general parameter values
    C_m, _, _, _, V_thresh, _, rate_window, E_e, E_i, tau_e, tau_i, Delta_T_ad, tau_w_ad, a_ad, b_ad, membrane_noise = sf.sim_paramters(parameter_mode='CTR', membrane_noise_mode=membrane_noise_mode)

    # unpack CTR & FR parameters
    
    R_m_CTR, R_m_FR = R_m[0], R_m[1]
    E_L_CTR, E_L_FR = E_L[0], E_L[1]
    r_post_CTR, r_post_FR = r_post[0], r_post[1]
    
    # calculate values for CTR
    # initialize w_scale for CTR
    w_scale_CTR = []
    r_post_simulation_CTR = []

    # iterate through all CTR values
    for R_m_exp, E_L_exp, r_post_exp in zip(R_m_CTR, E_L_CTR, r_post_CTR):
    
        g_L_exp = 10**(3)/R_m_exp # leak conductance in nS
        V_reset_exp = E_L_exp # reset potential in mV # -60 previously

        # initialize fitting parameters
        w_scale_init = 0.0001 # is the w_scale to start iteration with

        w_scale_iteration, r_post_simulation = single_w_scale_estimation_run(r_post_exp, w_scale_init, T, N_e, N_i, spike_times_e, spike_times_i, E_e, E_i, w_e_0, w_i_0, tau_e, tau_i, C_m, g_L_exp, E_L_exp, V_thresh, V_reset_exp, rate_window, Delta_T_ad, tau_w_ad, a_ad, b_ad, membrane_noise, model_mode, tolerance=0.0099, max_iterations=15)

        # save iteration results
        w_scale_CTR.append(w_scale_iteration)
        r_post_simulation_CTR.append(r_post_simulation)
        
    # calculate values for FR
    # initialize w_scale for FR
    w_scale_FR = []
    r_post_simulation_FR = []

    # iterate through all FR values
    for R_m_exp, E_L_exp, r_post_exp in zip(R_m_FR, E_L_FR, r_post_FR):
    
        g_L_exp = 10**(3)/R_m_exp # leak conductance in nS
        V_reset_exp = E_L_exp # reset potential in mV # -60 previously

        # initialize fitting parameters
        w_scale_init = 0.0005 # is the w_scale to start iteration with

        w_scale_iteration, r_post_simulation = single_w_scale_estimation_run(r_post_exp, w_scale_init, T, N_e, N_i, spike_times_e, spike_times_i, E_e, E_i, w_e_0, w_i_0, tau_e, tau_i, C_m, g_L_exp, E_L_exp, V_thresh, V_reset_exp, rate_window, Delta_T_ad, tau_w_ad, a_ad, b_ad, membrane_noise, model_mode, tolerance=0.0099, max_iterations=15)

        # save iteration results
        w_scale_FR.append(w_scale_iteration)
        r_post_simulation_FR.append(r_post_simulation)

    return w_scale_CTR, r_post_simulation_CTR, w_scale_FR, r_post_simulation_FR

################################ statistical test functions ################################

def Kolmogorov_Smirnov_test(dist_1, dist_2): 
    # sort the samples to prepare data for ECDFs 
    sample1 = np.sort(dist_1)
    sample2 = np.sort(dist_2)
    
    # get sample size
    n1 = len(sample1)
    n2 = len(sample2)
    
    # concatenate and sort all values
    all_values = np.sort(np.concatenate((sample1, sample2)))
    
    # compute empirical CDFs for both samples at each point
    ecdf1 = np.searchsorted(sample1, all_values, side='right') / n1
    ecdf2 = np.searchsorted(sample2, all_values, side='right') / n2
    
    # calculate the KS statistic with D being the maximum difference between CDFs
    D = np.max(np.abs(ecdf1 - ecdf2)) # if the distributions are identical, D will be close to 0; if they diverge, D increases
    

    # calculate p-value using Kolmogorov distribution approximation P(D > observed) ≈ 2 * sum_{j=1}^{∞} (-1)^{j-1} * exp(-2 * j^2 * λ^2)
    def ks_pvalue_approx(D, n1, n2):
        ne = n1 * n2 / (n1 + n2) # calculate effective sample size
        lambda_ = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * D  # asymptotic formula for large samples
        # This transforms D into a scaled version (λ) that matches the theoretical Kolmogorov distribution.
        # The extra constants (0.12 and 0.11/...) are correction terms for finite sample sizes.
        if lambda_ < 1e-5:
            return 1.0  # practically identical distributions
        # Kolmogorov approximation for large n
        p = 2 * sum((-1)**(j-1) * np.exp(-2 * (j**2) * lambda_**2) for j in range(1, 100))
        return min(max(p, 0.0), 1.0)

    D = np.max(np.abs(ecdf1 - ecdf2))
    p_value = ks_pvalue_approx(D, n1, n2)

    return D, p_value



