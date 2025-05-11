import mne
import os
import os.path as op
import numpy as np
import pandas as pd
mne.viz.set_3d_options(antialias=False)


rounds = [1, 2, 3, 4, 5]

trial_type = ['norisk',  'risk']
feedback = ['positive', 'negative']



labels =  mne.read_labels_from_annot("fsaverage", "HCPMMP1", hemi = "both")
labels = [lab for lab in labels if '???' not in lab.name]
label_names = [label.name for label in labels] 

data_path = "/media/kristina/storage/probability/sources/alpha_8_12/stc_by_epo_v3"

for subj in subjects:
    print(subj)
    df = pd.DataFrame()
    labels_morph= mne.morph_labels(labels,subject_to=subj,subject_from='fsaverage')
   
    src = mne.setup_source_space(subject =subj, spacing='oct6', add_dist=False,n_jobs=-1) # by default - spacing='oct6' (4098 sources per hemisphere)
    
    for r in rounds:
        for cond in trial_type:
            for fb in feedback:
                    
                try:
                    events_response = np.loadtxt('/media/kristina/storage/probability/events_trained_by_cond_WITH_mio_corrected/{0}_run{1}_{2}_fb_cur_{3}.txt'.format(subj, r, cond, fb), dtype='int')
                    # # если только одна метка, т.е. одна эпоха, то выдается ошибка, поэтому приводи shape к виду (N,3)
                    if events_response.shape == (3,):
                        events_response = events_response.reshape(1,3)
                    epochs_num = os.listdir(os.path.join(data_path, '{0}_run{1}_{2}_fb_cur_{3}'.format(subj, r, cond, fb)))
                    epo_n = (int(len(epochs_num) / 2))
                    #for ep in range(epo_n):
                    for ep, ev in zip(range(epo_n),events_response):
                        df_epo= pd.DataFrame()
                        print(ep)
                        time=ev[0,]
                        #sch = [s]*len(labels) 
                        stc = mne.read_source_estimate(os.path.join(data_path, '{0}_run{1}_{2}_fb_cur_{3}/{4}'.format(subj, r, cond, fb, ep)))
                        # morph = mne.compute_source_morph(stc, subject_from=subj, subject_to='fsaverage',src_to=fsaverage,verbose='error')
                           
                        # stc_fsaverage = morph.apply(stc)
                        stc2 = stc.copy()
                        stc2=stc2.crop(tmin=1.200, tmax=1.600, include_tmax=True) ### crop the time what you want to analyse

                        label_ts = mne.extract_label_time_course(stc2,labels_morph, src=src, mode='mean')
                        label_ts_avg=label_ts.mean(axis=1)
                        epo = [ep for i in range(len(labels))]
                        subject = [subj for i in range(len(labels))]
                        run = [r for i in range(len(labels))]
                        trial = [cond for i in range(len(labels))]
                        fb_cur=[fb for i in range(len(labels))]
                        #fb_prev= [feedback_prev for i in range(448)]
                        df_epo['time'] =[time for i in range(len(labels))]
                        df_epo['beta_power'] = label_ts_avg
                        df_epo['label'] = label_names
                        df_epo['epo'] = epo
                        df_epo['subject'] = subject
                        df_epo['round'] = run
                        df_epo['trial_type'] = trial
                        df_epo['feedback_cur'] = fb_cur
                        #df_epo['feedback_prev']  =  fb_prev
                        #df_epo['scheme']=sch
                        df = pd.concat([df,df_epo], axis=0)    
                         
                                       
                except (OSError, FileNotFoundError):
                    print('This file not exist')
    df.to_csv('/media/kristina/storage/probability/sources/alpha_8_12/df_1200_1600/{0}.csv'.format(subj))
                          
