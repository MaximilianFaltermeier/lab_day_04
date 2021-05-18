import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat


def nlms4echokomp(x, g, noise, alpha, mh):
    """ The MATLAB function 'nlms4echokomp' simulates a system for acoustic echo compensation using NLMS algorithm
    :param x:       Input speech signal from far speaker
    :param g:       impluse response of the simulated room
    :param noise:   Speech signal from the near speaker and the background noise(s + n)
    :param alpha:   Step size for the NLMS algorithm
    :param mh:      Length of the compensation filter

    :return s_diff:  relative system distance in dB
    :return err:    error signal e(k)
    :return x_hat:  output signal of the compensation filter
    :return x_tilde:acoustic echo of far speakers
    """

    # Initialization of all the variables
    lx = len(x)  # Length of the input sequence
    mg = len(g)  # Length of the room impulse response(RIR)
    if mh > mg:
        mh = mg
        import warnings
        warnings.warn('The compensation filter is shortened to fit the length of RIR!', UserWarning)

    # Vectors are initialized to zero vectors.
    x_tilde = np.zeros(lx - mg)
    x_hat = x_tilde.copy()
    err = x_tilde.copy()
    s_diff = x_tilde.copy()
    h = np.zeros(mh)

    # Realization of NLMS algorithm
    k = 0
    for index in range(mg, lx):
        # Extract the last mg values(including the current value) from the
        # input speech signal x, where x(i) represents the current value.
        x_block = x[lx - mg:]

        # Filtering the input speech signal using room impulse response and adaptive filter. Please note that you don't
        # need to implement the complete filtering here. A simple vector manipulation would be enough here
        x_tilde[k] = np.dot(g, x_block)
        x_hat[k] = -noise[k] - x_tilde[k]

        # Calculating the estimated error signal
        err[k] = x_tilde[k] - x_hat[k]

        # Updating the filter
        beta = alpha / np.dot(x_hat, x_hat)
        h = h + noise[k] * x_hat[k] * beta
        # Calculating the relative system distance
        d = g - h
        s_diff[k] = np.dot(d, d) / np.dot(g, g)

        k = k + 1  # time index

    s_diff = 10 * np.log10(s_diff[:k]).T

    # Calculating the relative system distance in dB
    return s_diff, err, x_hat, x_tilde


# switch between exercises
exercise = 5  # choose between 1-7

f = np.load('echocomp.npz')
g = [f['g1'], f['g2'], f['g3']]
s = f['s']

# Generation of default values
alpha = 0.1  # Step size for NLMS

ls = len(s)  # Length of the speech signal
n0 = np.sqrt(0.16) * np.random.randn(ls)  # White Noise
s = s / np.sqrt(s.T.dot(s)) * np.sqrt(n0.T.dot(n0))  # Number of curves in each plot (should not be changed)
vn = 3  # number of curves
noise = [np.zeros(ls, ) for i in range(vn)]  # no disturbance by noise
alphas = [alpha for i in range(vn)]  # Step size factor for different exercises
mh = len(g[0]) * np.ones(vn, dtype=int)  # Length of the compensation filter

x = [n0.copy() for i in range(vn)]  # white noise as input signal

# In the following part, the matrices and vectors must be adjusted to
# meet the requirement for different exercises
# (Exercise 1 can be simulated using only the initialized values above)

if exercise == 2:
    # Only the value of input speech signal need to be changed. All the other
    # vectors and parameters should not be modified

    x[0] = s  # Speech signal
    # white noise
    x[1] = np.random.normal(0, np.sqrt(0.16), size=22000)  # white noise
    x[2] = signal.lfilter([1], [1, -0.5], x[1])  # colorful noise
    leg = ('Speech', 'white noise', 'colorful noise')
    title = 'Different Input Signals'
elif exercise == 3:
    noise[0] = np.random.normal(0, 0, size=22000)
    noise[1] = np.random.normal(0, np.sqrt(0.001), size=22000)
    noise[2] = np.random.normal(0, np.sqrt(0.01), size=22000)
    leg = ('var = 0', 'var = 0.001', 'var = 0.01')
    title = 'Input Signals with different white background noise'
elif exercise == 4:
    # todo your code
    pass

elif exercise == 5:
    alphas = [0.1, 0.5, 1.0]
    noise[0] = np.random.normal(0, np.sqrt(0.01), size=22000)
    noise[1] = np.random.normal(0, np.sqrt(0.01), size=22000)
    noise[2] = np.random.normal(0, np.sqrt(0.01), size=22000)
    leg = ('alpha = 0.1', 'alpha = 0.5', 'alpha = 1')
    title = 'variation of the stepsize alpha'
elif exercise == 6:
    # todo your code
    pass

elif exercise == 7:
    # todo your code
    pass

# There should be appropriate legends and axis labels in each figure!
if exercise == 1:
    s_diff, e, x_h, x_t = nlms4echokomp(n0, g[0], np.zeros(ls), alpha, 200)
    erle = 10 * np.log10(np.dot(x_t, x_t) / (x_t - x_h) ** 2)
    fig, axs = plt.subplots(3)
    axs[0].plot(e, label='residual echo e')
    axs[0].plot(x_t, label='echo signal x_t')
    axs[1].plot(s_diff, label='relative system distance')
    axs[2].plot(erle, label='ERLE')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[1].set_ylabel('D in dB')
    axs[2].set_ylabel('ERLE in dB')
    axs[2].set_xlabel('k')
    plt.show()
else:
    for i in range(vn):
        # 3 system distances with different parameters are calculated here
        # The input variables of 'nlms4echokomp' must be adapted according
        # to different exercises.
        # TODO: g[0] -> g[i] fix problem of length
        s_diff, e, x_h, x_t = nlms4echokomp(x[i], g[0], noise[i], alphas[i], mh[i])
        plt.plot(s_diff, label=leg[i])

    plt.title('Exercise ' + str(exercise) + ': ' + title)
    plt.xlabel('k')
    plt.ylabel('D(k) [dB]')
    plt.grid(True)
    plt.legend()
    plt.show()
