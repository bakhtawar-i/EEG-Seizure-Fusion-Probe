"""
Builds an MNE Info object for CHB-MIT's 18 bipolar channels, with each
channel's spatial position approximated as its first electrode's standard
10-20 position (e.g. FP1-F7 -> position of FP1). This is a documented
approximation: bipolar (differential) signals don't have a true single-point
position, but this gives BENDR's channel-interpolation layer a reasonable
spatial anchor for each channel, consistent with the same first-electrode
approximation used for the LaBraM channel mapping in Phase 3.

Usage:
    from chbmit_chs_info import get_chbmit_chs_info
    chs_info = get_chbmit_chs_info()
"""
import mne

BIPOLAR_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
]


def get_chbmit_chs_info(sfreq=256):
    info = mne.create_info(ch_names=BIPOLAR_CHANNELS, sfreq=sfreq, ch_types='eeg')

    montage = mne.channels.make_standard_montage('standard_1020')
    montage_pos = montage.get_positions()['ch_pos']

    pos_dict = {}
    for bipolar_name in BIPOLAR_CHANNELS:
        electrode = bipolar_name.split('-')[0]
        matched_key = next(k for k in montage_pos.keys() if k.upper() == electrode.upper())
        pos_dict[bipolar_name] = montage_pos[matched_key]

    custom_montage = mne.channels.make_dig_montage(ch_pos=pos_dict, coord_frame='head')
    info.set_montage(custom_montage)

    return info['chs']


if __name__ == "__main__":
    chs_info = get_chbmit_chs_info()
    print(f"Total channels: {len(chs_info)}")
    print(chs_info[0])


# Total channels: 18
# {'loc': array([-0.0294367,  0.0839171, -0.00699  ,  0.       ,  0.       ,
#         0.       ,        nan,        nan,        nan,        nan,
#               nan,        nan]), 'unit_mul': 0 (FIFF_UNITM_NONE), 'range': 1.0, 'cal': 1.0, 'kind': 2 (FIFFV_EEG_CH), 'coil_type': 1 (FIFFV_COIL_EEG), 'unit': 107 (FIFF_UNIT_V), 'coord_frame': 4 (FIFFV_COORD_HEAD), 'ch_name': 'FP1-F7', 'scanno': 1, 'logno': 1}
