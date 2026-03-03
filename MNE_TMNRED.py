import mne
import numpy as np
import pandas as pd
import os
from mne_bids import BIDSPath, read_raw_bids
from mne.time_frequency import psd_array_multitaper

subjects = ["01", "02", "03", "04",
            "05", "06","07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30"
            ]

sessions = ["1", "2", "3", "4", "5", "6", "7", "8"]

path = "path-to-BIDS-dataset"
output_dir = "output-directory-path"

for sub in subjects:
    all_sessions_dfs = []
    for ses in sessions:
        bids_path = BIDSPath(
            root=path,
            subject=sub,
            session=ses,
            task="fuzzysemanticrecognition",
            datatype="eeg"
            )

        raw=read_raw_bids(bids_path)  
        raw.load_data() 
        raw.pick_types(eeg=True, eog=False, ecg=False, misc=False, stim=False)

        # If montage not specified use standard 10-20 montage
        if raw.get_montage() is None:
            raw.set_montage("standard_1020", on_missing="ignore")


        # Plot raw data and raw info
        print(f"raw info and plots for sub-{sub}")
        print(raw.info)
        print("Raw data plots:")
        # raw.plot()
        # raw.plot_psd(average=True)


        # ---------------- BAD CHANNELS DETECTION (Manual) ----------------
        #if len(raw.info["bads"]) > 0:
        #    print("These are the channels already marked as bad:")
        #    print(raw.info["bads"])

        #bad_chs = input(
        #    "Type the channels you want to mark as bad separated by a comma (es. PO8,F7) then press return:"
        #)

        #if bad_chs.strip():
        #    new_bads = [ch.strip() for ch in bad_chs.split(",")]
        #    raw.info["bads"] = list(set(raw.info["bads"] + new_bads))


        # ---------------- BAD CHANNEL DETECTION (Automatic) ----------------
        spectrum = raw.compute_psd(
            method='welch',
            fmin=1,
            fmax=40,
            n_fft=2048,
            picks='eeg',
            verbose=False
        )
        psds = spectrum.get_data()
        freqs = spectrum.freqs


        mean_power = psds.mean(axis=1)


        threshold = mean_power.mean() + 2 * mean_power.std()


        bad_channels_indices = np.where(mean_power > threshold)[0]
        bad_channels = np.array(raw.ch_names)[bad_channels_indices]

        raw.info["bads"] = list(bad_channels)

        print("Automatically detected bad channels:", raw.info["bads"])

        #save bad channels
        n_bad_channels = len(bad_channels)
        bad_channels_str = ",".join(bad_channels) if n_bad_channels > 0 else "None"
        total_channels = len(raw.ch_names)
        perc_bad_channels = (n_bad_channels / total_channels) * 100
            


        # ---------------- INTERPOLATION ----------------
        if len(raw.info["bads"]) > 0:
            raw.interpolate_bads(reset_bads=True)
        
        print("Interpolation completed")
        

        # ---------------- RE-REFERENCING ----------------
        raw.set_eeg_reference("average")
        print("Re-referencing completed")


        # ---------------- FILTERING ----------------
        raw.filter(0.5, 80, fir_design="firwin")
        raw.notch_filter(freqs=50)
        print("Filtering completed")


        # ---------------- DOWNSAMPLING ----------------
        raw.resample(200)
        print("downsampling completed")


        # ---------------- ICA ----------------
        ica = mne.preprocessing.ICA(n_components=0.99, random_state=0)
        raw_ica = raw.copy().filter(1, None)
        ica.fit(raw_ica)
        
        print("Look at the plot for ica to exclude:")
        ica.plot_components() 
        bad_ica = input(
            "Type the ica indexe to exclude, separated by a comma (es. 0,2,5) then press return:"
        )
        if bad_ica.strip():
            ica.exclude = [int(ic) for ic in bad_ica.split(",")]

        raw = ica.apply(raw)

        print("ICA completed", f"removed ICA number: {bad_ica}")
        

        # ---------------- EPOCHING ----------------
        events, event_id = mne.events_from_annotations(raw)

        t_min= 0
        t_max = 2
        s_epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=t_min,
            tmax=t_max,
            baseline=None,
            preload=True
        )

        data = s_epochs.get_data()  # shape: (n_trials, n_channels, n_times)

        # Metric 1: median variance per trial 
        var_per_trial = np.var(data, axis=2)  # variance per channel
        metric1 = np.median(var_per_trial, axis=1)

        # Metric 2: median deviation from mean 
        mean_per_trial = np.mean(data, axis=2, keepdims=True)
        diff = np.abs(data - mean_per_trial)
        metric2 = np.median(diff.mean(axis=2), axis=1)

        # Thresholding 
        thr1 = metric1.mean() + 2 * metric1.std()
        thr2 = metric2.mean() + 2 * metric2.std()

        bad_trials_1 = np.where(metric1 > thr1)[0]
        bad_trials_2 = np.where(metric2 > thr2)[0]

        bad_trials = np.union1d(bad_trials_1, bad_trials_2)

        print("Rejected trials:", bad_trials)

        # save bad trials
        n_bad_epochs = len(bad_trials)
        bad_epochs_str = ",".join(map(str, bad_trials)) if n_bad_epochs > 0 else "None"
        total_epochs_before = len(metric1)
        perc_bad_epochs = (n_bad_epochs / total_epochs_before) * 100

        # drop bad trials
        s_epochs.drop(bad_trials)

        print("Epoching completed")


        # ---------------- COMPUTE EEG DATA----------------
        # Select channels
            # All available = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 
            #                   'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2', 'Status']
        picks = ['Fz','F3','F4','Cz','C3','C4','Pz','P3','P4','Oz','O1','O2']
        s_epochs.pick(picks)
        print(f"Channels picked: {picks}")

        # EEG features
        s_epochs_late = s_epochs.copy().crop(0.5,2)
        s_epochs_early = s_epochs.copy().crop(0,0.5)
        data_late = s_epochs_late.get_data()
        data_early = s_epochs_early.get_data()

        amp_average_late = np.mean(data_late, axis=2)
        amp_average_early = np.mean(data_early, axis=2)

        # PSD per channel per sentence
        bands = {'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,40)}
        psd_per_s_late = []
        psd_per_s_early = []

        for e1 in s_epochs_late:
            psd, freqs = psd_array_multitaper(e1, sfreq=raw.info['sfreq'], fmin=4, fmax=40, verbose=False)
            bp = {band: psd[:, (freqs>=fmin) & (freqs<fmax)].mean(axis=1)
                  for band, (fmin,fmax) in bands.items()}
            psd_per_s_late.append(bp)
        
        for e2 in s_epochs_early:
            psd, freqs = psd_array_multitaper(e2, sfreq=raw.info['sfreq'], fmin=4, fmax=40, verbose=False)
            bp = {band: psd[:, (freqs>=fmin) & (freqs<fmax)].mean(axis=1)
                  for band, (fmin,fmax) in bands.items()}
            psd_per_s_early.append(bp)


        # plot
        print(f"Displaying plots for sub-{sub}, ses-{ses}")
        print(s_epochs.info)
        # s_epochs.plot()
        # s_epochs.compute_psd().plot_topomap(bands=bands)


        # ---------------- EEG DATA ----------------
        labels = s_epochs.events[:,-1]
        features = []
        for i in range(len(s_epochs)):
            row = {}

            row["sub"] = sub
            row["session"] = ses
            for ev in event_id:
                if event_id[ev] == labels[i]:
                    row["label"] = ev
            
            for ch_indx1, val2 in enumerate(amp_average_late[i]):
                row[f"amp_late_ch{picks[ch_indx1]}"] = val2
            
            for ch_indx2, val3 in enumerate(amp_average_early[i]):
                row[f"amp_early_ch{picks[ch_indx2]}"] = val3
            
            for band in bands:
                band_power_late = psd_per_s_late[i][band]
                for ch_indx3, val4 in enumerate(band_power_late):
                    row[f"{band}_late_ch{picks[ch_indx3]}"] = val4
                row[f"{band}_late_global"] = band_power_late.mean()

                band_power_early = psd_per_s_early[i][band]
                for ch_indx4, val5 in enumerate(band_power_early):
                    row[f"{band}_early_ch{picks[ch_indx4]}"] = val5
                row[f"{band}_early_global"] = band_power_early.mean()
            
            features.append(row)

        df_s = pd.DataFrame(features)
        all_sessions_dfs.append(df_s)
        print(f"Session {ses} processed for subject {sub}")


        # ---------------- QC REPORT ----------------
        qc_info = {
            "sub": sub,
            "session": ses,
            "n_bad_channels": n_bad_channels,
            "perc_bad_channels": perc_bad_channels,
            "bad_channels": bad_channels_str,
            "n_bad_epochs": n_bad_epochs,
            "perc_bad_epochs": perc_bad_epochs,
            "bad_epochs_indices": bad_epochs_str
        }

        df_qc = pd.DataFrame([qc_info])
        df_qc.to_csv(os.path.join(output_dir, f"sub-{sub}_ses-{ses}_QC_report.csv"), index=False)

        print(f"QC report saved: sub-{sub}_ses-{ses}_QC_report.csv")

    df_sub = pd.concat(all_sessions_dfs, ignore_index=True)
    df_sub.to_csv(os.path.join(output_dir,f"sub-{sub}_ALL_sessions_features.csv"), index=False)

    print(f"All sessions saved in one file: sub-{sub}_ALL_sessions_features.csv")


print("All subjects completed")
