import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn import neighbors
from sklearn.cross_validation import train_test_split

data = np.load('motor_dataset.npy')[()]
trial_data = pd.DataFrame({'angle': data['trial_angle'],
                           'go': data['trial_go'],
                           'move': data['trial_move'],
                           'acq': data['trial_acq']})
trial_data['time_from_go'] = trial_data.acq - trial_data.go
trial_data['time_from_move'] = trial_data.acq - trial_data.move

def spikes_between(start, end, norm=True):
    """Returns the number of spikes in each channel that are in the given range"""
    in_range = []
    for times in data['spk_times']:
        spikes = times[(times >= start) & (times <= end)]
        res = len(spikes)/((end-start) if norm else 1)
        in_range.append(res)
    return in_range
    
trial_data['spks_in_move'] = trial_data.apply(
    lambda r: spikes_between(r.move, r.acq), axis=1)
trial_data['spks_total'] = trial_data.apply(
    lambda r: spikes_between(r.move, r.acq), axis=1)

angles = trial_data.angle.tolist()
spks = trial_data.spks_in_move.tolist()

scores = []
for i in range(20):
    angles_train, angles_test, spks_train, spks_test = train_test_split(angles, spks)

    clf = neighbors.KNeighborsClassifier(n_neighbors=10)
    clf.fit(spks_train, angles_train)

    scores.append(clf.score(spks_test, angles_test))

print(sum(scores)/len(scores))