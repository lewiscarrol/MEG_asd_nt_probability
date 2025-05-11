

import mne
import os
import os.path as op
import numpy as np
import pandas as pd
from multiprocessing import Pool
mne.viz.set_3d_options(antialias=False)

os.environ['SUBJECTS_DIR'] = '/media/kristina/storage/probability/freesurfer'
subjects_dir = '/media/kristina/storage/probability/freesurfer'

trial_type = ['norisk', 'risk']


feedback = ['positive', 'negative']

#### setup freq bands######
L_freq = 10
H_freq = 31
f_step = 2

freqs = np.arange(L_freq, H_freq, f_step)
n_cycles = freqs//2

period_start = -1.750
period_end = 2.750

baseline = (-0.35, -0.05)


rounds = [1, 2, 3, 4, 5]


for subj in subjects:
    inv = mne.minimum_norm.read_inverse_operator('/media/kristina/storage/probability/sources/inverse_operator/{0}-inv.fif'.format(subj)) 
    
    for cond in trial_type:
        for fb in feedback:
            for r in rounds:
                try:
                    #######events for baseline######
                    events_list =[]
                    events_pos = np.loadtxt("/media/kristina/storage/probability/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_positive_fix_cross.txt".format(subj, r), dtype='int') 

                        # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
                    if events_pos.shape == (3,):
                        events_pos = events_pos.reshape(1,3)
                        
                    events_list.append(events_pos)
                    
                    events_neg = np.loadtxt("/media/kristina/storage/probability/fix_cross_mio_corr/{0}_run{1}_norisk_fb_cur_negative_fix_cross.txt".format(subj, r), dtype='int')
                    
                    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводим shape к виду (N,3)
                    if events_neg.shape == (3,):
                        events_neg = events_neg.reshape(1,3) 
                    events_list.append(events_neg)
                    if len(events_list) >0:
                        events=np.vstack(events_list)
                        events=np.sort(events,axis=0)
                        
                    else:
                       print('no events for baseline')
                    
                    #events, which we need
                    events_response = np.loadtxt('/media/kristina/storage/probability/events_trained_by_cond_WITH_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, cond, fb), dtype='int')
                    # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводи shape к виду (N,3)
                    if events_response.shape == (3,):
                        events_response = events_response.reshape(1,3)

                    raw_fname = op.join(data_path, '{0}/run{1}_{0}_raw_ica.fif'.format(subj, r))

                    raw_data = mne.io.Raw(raw_fname, preload=True)
                    
                    picks = mne.pick_types(raw_data.info, meg = True, eog = True,exclude='bads')
                    
                    epochs_bl = mne.Epochs(raw_data, events, event_id = None, tmin = -1.0, tmax = 1, baseline = None, picks = picks, preload = True)
                    #epochs_bl.resample(100)
                    freq_show_baseline = mne.time_frequency.tfr_morlet(epochs_bl, freqs = freqs, n_cycles = n_cycles, output='complex',decim=5, use_fft = False, return_itc = False, average=False).crop(tmin=baseline[0], tmax=baseline[1], include_tmax=True) #frequency of baseline
                    stc_bl= mne.minimum_norm.apply_inverse_tfr_epochs(freq_show_baseline, inv,  lambda2=1/1**2,method='sLORETA', prepared=False, method_params=None, use_cps=True, verbose=None)
                    
                    sum_array_bl = [np.array([array.data for array in array_bl]) for array_bl in stc_bl]
                    sum_freq_bl = np.array(sum_array_bl).sum(axis=0)
                    sum_freq_bl = 10*np.log10(sum_freq_bl)
                    
                    bl = np.mean(sum_freq_bl, axis=(0,2))
                    
                    epochs = mne.Epochs(raw_data, events_response, tmin=period_start, tmax=period_end, baseline=None, picks=picks, preload=True)
                    #epochs.resample(100)
                    freq_show = mne.time_frequency.tfr_morlet(epochs, freqs = freqs, n_cycles =n_cycles, output='complex',decim=5, use_fft = False, return_itc = False, average=False)
                    
                    stc= mne.minimum_norm.apply_inverse_tfr_epochs(freq_show, inv,  lambda2=1/1**2,method='sLORETA', prepared=False, method_params=None, use_cps=True, verbose=None)
                    
                    sum_array= [np.array([array.data for array in arrays]) for arrays in stc]
                    temp = stc[0][0]
                    sum_freq = np.array(sum_array).sum(axis=0)
                    os.makedirs('/media/kristina/storage/probability/sources/freq_10_31/stc_by_epo_v3/{0}_run{1}_{2}_fb_cur_{3}'.format(subj, r, cond, fb),exist_ok=True)
                    
                    tfr_after_bl = 10*np.log10(sum_freq) - bl[np.newaxis,:,np.newaxis]
                    for t in range(len(tfr_after_bl)):
                        temp.data =tfr_after_bl[t,:,:]
                        temp.save('/media/kristina/storage/probability/sources/freq_10_31/stc_by_epo_v3/{0}_run{1}_{2}_fb_cur_{3}/{4}'.format(subj, r, cond, fb,t),overwrite=True)

                except (OSError):
                    print('This file not exist')
                   
