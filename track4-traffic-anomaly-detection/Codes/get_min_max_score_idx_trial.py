from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        print("smooth only accepts 1 dimension arrays.")
        assert 1 == 2

    if x.size < window_len:
        print("Input vector needs to be bigger than window size.")
        assert 1 == 2

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        print("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
        assert 1 == 2

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def plot(vid, ypoints):
    fig = plt.figure(figsize = (5.15,5.15))
    plt.figtext(0, 0,str(vid))
    plt.subplot(111)
    y = smooth(ypoints, 10, 'bartlett')
# 1 (85.53333333333333, 1.0, 25.53333333333333)
# 1 (66.63333333333334, 1.0, 6.63333333333334)
# 32 (751.3333333333333, 12.0, 31.333333333333258)
# 32 (751.3, 12.0, 31.299999999999955)
# 35 (271.7, 4.0, 31.69999999999999)
# 35 (271.7, 4.0, 31.69999999999999)
# 46 (114.56666666666668, 1.0, 54.56666666666668)
# 46 (114.63333333333334, 1.0, 54.63333333333334)
# 64 (506.09999999999997, 8.0, 26.099999999999966)
# 64 (506.06666666666666, 8.0, 26.066666666666663)
# 68 (155.06666666666666, 2.0, 35.06666666666666)
# 68 (155.06666666666666, 2.0, 35.06666666666666)
# 82 (82.33333333333333, 1.0, 22.33333333333333)
# 82 (82.46666666666667, 1.0, 22.46666666666667)

#     y = smooth(ypoints, 50, 'hanning')
    
    plt.plot(ypoints)
    plt.plot(y)
    plt.savefig('plot_' + str(vid) + '.png')
    return y
    
video_names = [1, 32, 35, 46, 64, 68, 82]
# video_names = [68]
for video_name in video_names:
    file_name = '../images/ai_city_' + str(video_name) + '/scores.txt'
#     print(file_name)
    f = open(file_name, 'r').readlines()
    results = np.array(list(map(float, f))[:-1])
#     results = list(map(float, f))
    y = plot(video_name, results)
    
    spare_length = (len(y) - len(results) - 1)//2
    y = y[spare_length:-spare_length]
    
    idx_min = np.argmin(results)
    idx_max = np.argmax(results)
    idx = np.argmin(y)
    print(video_name, np.argmin(y), np.argmax(y))
    print(video_name, idx_min, idx_max)
#     print(video_name, idx + np.argmax(results[idx-3:idx]) - 1, idx_max)
#     print(video_name, np.argmin(y) + 1)
#     print(video_name, idx_min)
#     print(video_name, idx + np.argmax(results[idx-3:idx]) - 1)
#     break
    