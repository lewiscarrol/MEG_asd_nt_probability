import mne
import os
import os.path as op
import numpy as np
import pandas as pd
mne.viz.set_3d_options(antialias=False)


for subj in subjects:
    bem = mne.read_bem_solution('/media/kristina/storage/probability/bem/{0}_bem.h5'.format(subj), verbose=None)
    
    #
    src = mne.setup_source_space(subject =subj, spacing='oct6', add_dist=False,n_jobs=-1) # by default - spacing='oct6' (4098 sources per hemisphere)
    print(subj)
    raw_er = mne.io.read_raw_fif('/media/kristina/storage/probability/empty_room/{0}_er_raw_bads.fif'.format(subj),preload=True) 
    
    raw_er.notch_filter(np.arange(50, 201, 50), filter_length='auto', phase='zero', n_jobs=-1)
    raw_er.filter(1, 40, fir_design='firwin')

    picks_meg = mne.pick_types(raw_er.info, meg=True, eeg=False, eog=False, stim=False, exclude=['bads'])
    rank=mne.compute_rank(raw_er,rank='info', info=raw_er.info)
    cov = mne.compute_raw_covariance(raw_er,  method='shrunk',picks=picks_meg,rank=rank, n_jobs=-1)
    cov = mne.cov.regularize(cov, raw_er.info,  grad=0.1,mag=0.1)
    rank=mne.compute_rank(cov,rank='info', info=raw_er.info)
    trans = '/media/kristina/storage/probability/freesurfer/{0}/mri/T1-neuromag/sets/{0}-COR.fif'.format(subj)
    #trans='fsaverage'  
    raw_fname = op.join(data_path, '{0}/run1_{0}_raw_ica.fif'.format(subj))

    raw_data = mne.io.Raw(raw_fname, preload=True)
    fwd = mne.make_forward_solution(info=raw_data.info, trans=trans, src=src, bem=bem,n_jobs=-1)	                
    inv = mne.minimum_norm.make_inverse_operator(raw_data.info, fwd, cov, loose=0.2,depth=0.8, rank=rank)
    mne.minimum_norm.write_inverse_operator('/media/kristina/storage/probability/sources/inverse_operator/{0}-inv.fif'.format(subj),inv,overwrite=True)
    
    
    
