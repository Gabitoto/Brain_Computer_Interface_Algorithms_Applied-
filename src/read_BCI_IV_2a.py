import numpy as np
import scipy.io as sio
import scipy.signal as signal
from mne.io import read_raw_gdf
from mne import events_from_annotations
from modules.preprocessing.filtering import get_bandpass_coef
import glob
import logging


def read_BCI_IV_DS2a(filename, class_labels=[1, 2, 3, 4], band_pass=None, segment_length=2, offset=0.5, debug=False, selected_channels=None):
    
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

    n_subjects = 9
    fs = 250
    n_channels = 22
    segment_samples = segment_length * fs
    offset_samples = offset * fs
    # from the pdf file
    stim_codes = [769, 770, 771, 772] # 769: left hand, 770: right hand, 771: feet, 772: tongue - Cue onsets
    # keep only the trials from the selected classes, if needed
    stim_codes = [stim_codes[i-1] for i in class_labels] # as example [769, 770] for left and right hand
    logging.debug(f"stim_codes: {stim_codes}")

    # ['EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 'EEG-C3', 'EEG-6', 'EEG-Cz', 
    # 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 
    # 'EEG-15', 'EEG-16', 'EOG-left', 'EOG-central', 'EOG-right']
    channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
                   'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 
                   'Pz', 'P2', 'POz']
    
        
    train_eeg_files = sorted( glob.glob(filename + '**/*T.gdf', recursive=True) )
    logging.debug(f"Number of training files found: {len(train_eeg_files)}")
    test_eeg_files = sorted( glob.glob(filename + '**/*E.gdf', recursive=True) )
    logging.debug(f"Number of testing files found: {len(test_eeg_files)}")
    train_label_files = sorted( glob.glob(filename + '**/*T.mat', recursive=True) )
    logging.debug(f"Number of training label files found: {len(train_label_files)}")
    test_label_files = sorted( glob.glob(filename + '**/*E.mat', recursive=True ) )
    logging.debug(f"Number of testing label files found: {len(test_label_files)}")

    subject_data = {}

    for subject in range(n_subjects):
        
        # read the training data
        train_raw = read_raw_gdf(train_eeg_files[subject], preload=True, verbose=False)
        test_raw = read_raw_gdf(test_eeg_files[subject], preload=True, verbose=False)
        logging.debug(f"channels: {train_raw.ch_names}")
        
        if selected_channels is not None:
            channel_name_map = {
                nombre: real
                for nombre, real in zip(channel_list, train_raw.ch_names[:22])
            }

            try:
                selected_real_names = [channel_name_map[ch] for ch in selected_channels]
            except KeyError as e:
                raise ValueError(f"Canal {e} no encontrado en el mapeo. Verificá los nombres válidos: {list(channel_name_map.keys())}")
    
            logging.debug(f"Canales solicitados: {selected_channels}")
            logging.debug(f"Canales reales usados: {selected_real_names}")
    

            train_raw.pick_channels(selected_real_names)
            test_raw.pick_channels(selected_real_names)
            
            train_raw_data = train_raw.get_data()
            test_raw_data = test_raw.get_data()
            n_channels = len(selected_real_names)
        else:
            train_raw_data = train_raw.get_data()[:n_channels, :]
            test_raw_data = test_raw.get_data()[:n_channels, :]

        logging.debug(f"train_raw_data shape: {train_raw_data.shape}")
        logging.debug(f"test_raw_data shape: {test_raw_data.shape}")
        
        if band_pass:
            # this dataset is already filtered at 0.5 and 100 Hz and notch filtered at 50 Hz
            b,a = get_bandpass_coef(N=4, low=band_pass[0], high=band_pass[1], fs=fs)
            train_raw_data = signal.filtfilt(b, a, train_raw_data, axis=1)
            test_raw_data = signal.filtfilt(b, a, test_raw_data, axis=1)

        events, events_id = events_from_annotations(train_raw) 
        # events_id is a dict with the event codes {'1023':1, '769':2, '770':3, '771':4, '772':5}
        # get the indices of the events, event_index will be [2, 3] for left and right hand
        train_index_codes = [events_id[str(i)] for i in stim_codes] # [2, 3]
        logging.debug(f"train_index_codes: {train_index_codes}")
        #there are 48 trials for each class in a run (12 trials for each of the 4 possible classes)
        #each session has 6 runs, so 288 trials per session
        #'768' is the start of a run
        train_labels = sio.loadmat(train_label_files[subject], squeeze_me=True)['classlabel']
        train_labels = np.array([i for i in train_labels if i in class_labels])
        logging.debug(f"Number of train labels: {len(train_labels)}")
        logging.debug(f"train labels: {train_labels}")
        n_train_trials = np.sum( np.isin(events[:,-1], train_index_codes) )
        if n_train_trials != len(train_labels):
            raise ValueError(f"Number of trials in the labels file ({len(train_labels)}) does not match the number of trials in the data ({n_train_trials})")
        
        #[Ns * Nc * Nt]
        train_data = np.zeros(( segment_samples, n_channels, n_train_trials))
        training_labels = np.zeros((n_train_trials,))
        key_list = list(events_id.keys())
        val_list = list(events_id.values())
        i = 0
        for e in events:             
            if e[-1] in train_index_codes:
                code = int( key_list[ val_list.index(e[-1]) ] )
                start = int(e[0] + offset_samples)
                end = int(e[0] + offset_samples + segment_samples)
                train_data[:,:,i] = train_raw_data[:, start:end].T
                training_labels[i] = int(code - 768)
                if i%48 == 0:
                    logging.debug(f"code: {code}")
                    logging.debug(f"event: {e[-1]}")
                i += 1

        logging.debug(f"training labels: {training_labels}")
        # compare training_labels with train_labels
        if not np.all(training_labels == train_labels):
            raise ValueError("Labels do not match")
        
        events, events_id = events_from_annotations(test_raw)
        test_labels = sio.loadmat(test_label_files[subject], squeeze_me=True)['classlabel']
        test_mask = np.isin(test_labels, class_labels)
        
        n_test_trials = test_labels.size #np.sum(test_mask)
        
        test_data = np.zeros((segment_samples, n_channels, n_test_trials))
        i = 0
        for e in events:            
            if e[-1] == events_id['783']:#cue onset unknown class
                start = int(e[0] + offset_samples)
                end = int(e[0] + offset_samples + segment_samples)
                test_data[:,:,i] = test_raw_data[:, start:end].T
                i += 1
        
        testing_labels = test_labels[test_mask]
        test_data = test_data[:,:,test_mask]

        subject_data[f'subject_{subject+1}'] = {
                                                'X_train': train_data, 
                                                'y_train': training_labels, 
                                                'X_test': test_data, 
                                                'y_test': testing_labels
                                                }
        
    return subject_data


if __name__=='__main__':

    filename = './data/BCI_IV_2a/'
    data = read_BCI_IV_DS2a(filename, class_labels=[1, 2], band_pass=[8, 30], segment_length=2, offset=0.5, debug=True, selected_channels= ['FC3', 'C5', 'C3', 'C1', 'CP3'])

    np.save('./data/BCI_IV_2a/segmentadas/BCI_IV_2a_right_upper_limb_8_30.npy', data)

    print(data.keys())
    print(data['subject_1']['X_train'].shape)
    print(data['subject_1']['y_train'].shape)
    print(data['subject_1']['X_test'].shape)
    print(data['subject_1']['y_test'].shape)







