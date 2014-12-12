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
trial_data['avg_spk_rate'] = trial_data.apply(
    lambda r: sum(r.spks_in_move)/len(r.spks_in_move), axis=1)

angles = trial_data.angle.tolist()
spks_in_move = trial_data.spks_in_move.tolist()
spks_total = trial_data.spks_in_move.tolist()

def score(classifier, test_spks, test_angles):
    """returns a 4 percentages.  The first is the percent exactly right
    The second is the percent no more than 45 degrees off, the third no more
    than 90 degrees off, and the last no more than 135 degrees off.
    """
    n = len(test_spks)
    totals = [n, n, n, n, n]
    for result, actual in zip(classifier.predict(test_spks), test_angles):
        dst = abs(result - actual)
        if dst > 180:
            dst = 360 - dst
        for i in range(dst/45):
            totals[i] -= 1.
    return [t/len(test_spks) for t in totals]

def test_predictor(k=12, n=100, weights='distance', spks=spks_in_move):
    """Runs the predictor multiple times and returns how often it gets results
    correct with an increasing level of precision.
    The first number is for perfect results, second for within 45 degrees,
    thrid for within 90 degrees, etc.
    """
    scores = []
    for i in range(n):
        angles_train, angles_test, spks_train, spks_test = train_test_split(angles, spks)

        clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights=weights)
        clf.fit(spks_train, angles_train)

        scores.append(score(clf, spks_test, angles_test))

    return [sum(map(lambda x: x[i], scores))/len(scores) for i in range(5)]
    
def plot_speed_spks(data, angle=None, title=None, ploton=plt):
    relevant = data[data.angle == angle] if angle != None else data
    xs = relevant.time_from_move
    ys = relevant.avg_spk_rate
    polynomial = np.polyfit(xs, ys, 1)
    fit = np.poly1d(polynomial)
    ploton.plot(xs, ys, 'o', [0,.5], fit([0,.5]), 'b-')
    if ploton != plt:
        ploton.set_title(title)
        ploton.set_xlim([min(xs), max(xs)])
        ploton.set_xlabel("Time in seconds moving")
        ploton.set_ylabel("Average Spike rate")
    
def graph_speed_plots(data):
    f, axis = plt.subplots(2,4)
    for i in range(8):
        angle = i * 45
        plot_speed_spks(data, angle=angle, 
                        title = "Trials with direction %i degrees"%angle,
                        ploton = axis[i%2][i/2])
    f.show()