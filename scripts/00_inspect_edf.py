import mne

raw = mne.io.read_raw_edf('data/raw/chbmit/chb01/chb01_03.edf', preload=True)
print(raw.info)
print('Channels:', raw.ch_names)
print('Sampling rate:', raw.info['sfreq'])
print('Duration (s):', raw.times[-1])


# *** OUTPUT ***
# <Info | 8 non-empty values
#  bads: []
#  ch_names: FP1-F7, F7-T7, T7-P7, P7-O1, FP1-F3, F3-C3, C3-P3, P3-O1, ...
#  chs: 23 EEG
#  custom_ref_applied: False
#  highpass: 0.0 Hz
#  lowpass: 128.0 Hz
#  meas_date: 2076-11-06 13:43:04 UTC
#  nchan: 23
#  projs: []
#  sfreq: 256.0 Hz
#  subject_info: <subject_info | his_id: Surrogate>
# >
# Channels: ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8-0', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8-1']
# Sampling rate: 256.0
# Duration (s): 3599.99609375

# -> That confirms it: 256 Hz sampling rate, 23 bipolar montage channels, and the duplicate T8-P8 channel MNE just handled by renaming to T8-P8-0/T8-P8-1. 
# -> Good catch by the warning — this is a known CHB-MIT quirk (a few patients have slightly different channel configurations or naming), so worth handling deliberately rather than assuming it's uniform.

# -> Since channel sets vary slightly across patients, picking a common channel subset up front 
# -> the standard approach is to use the 18–23 channels that appear across all/most patients and drop or ignore any patient-specific extras (like this duplicate T8-P8). 
# -> This avoids silent shape mismatches later when you batch across patients. 
# -> Worth checking chb01's channel list against chb02's before finalizing which channels to standardize on — CHB-MIT's known inconsistency is exactly the kind of thing that causes a "not applicable to some patients" bug if we dont check now.