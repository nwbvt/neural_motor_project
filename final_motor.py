import numpy as np
import matplotlib.pylab as plt
import pandas as pd

data = np.load('motor_dataset.npy')[()]
trial_data = pd.DataFrame({'angle': data['trial_angle'],
                           'go': data['trial_go'],
                           'move': data['trial_move'],
                           'acq': data['trial_acq']})
trial_data['time_from_go'] = trial_data.acq - trial_data.go
trial_data['time_from_move'] = trial_data.acq - trial_data.move

num_spikes = len(data['spk_channels'])
spk_data = [(data['spk_channels'][i], data['spk_times'][i])
             for i in range(num_spikes)]

def spikes_between(start, end, norm=True):
    in_range = []
    for channel, times in spk_data:
        spikes = times[(times >= start) & (times <= end)]
        res = len(spikes)/((end-start) if norm else 1)
        in_range.append((channel, res))
    return in_range
    
trial_data['spks_in_move'] = trial_data.apply(
    lambda r: spikes_between(r.move, r.acq), axis=1)
trial_data['spks_total'] = trial_data.apply(
    lambda r: spikes_between(r.move, r.acq), axis=1)
