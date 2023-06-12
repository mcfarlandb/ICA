import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

ica = FastICA(whiten='arbitrary-variance')  # Create ICA object

def restrictAngle(angle):
    if angle < -45:
        angle = angle + 180
    if angle > 135:
        angle = angle - 180
    return angle


# computes dfts of multiple signals in a matrix
def fft_matrix(x, T):
    # Compute FFT of all signals in the matrix x, each row is a signal.
    X = np.zeros(x.shape)
    N = x.shape[1]

    for i in range(x.shape[0]):
        X[i] = np.abs(np.fft.fft(x[i])) / N

    omega_k = np.arange(N) / T  # This is the list of frequencies
    return X, omega_k


def icaVariables(s, MixFirstAngle, MixLastAngle, theta, i):
    M = np.array([[math.cos(math.radians(MixFirstAngle)), math.cos(math.radians(MixLastAngle)),
                   math.cos(math.radians(theta[i]))],
                  [math.sin(math.radians(MixFirstAngle)), math.sin(math.radians(MixLastAngle)),
                   math.sin(math.radians(theta[i]))]])

    # Create mixed matrix
    x = M @ s  # matrix multiplication of M and s

    # Create ICA unmixing matrix and components

    ica.fit(x.T)  # Create unmixing matric
    U = ica.components_  # Set unmixing matrix to U
    M_ICA = np.linalg.inv(U)  # Take inverse of unmixing matrix to find ICA mixing matrix
    c = U @ x

    return M, M_ICA, x, c


# returns relative amplitude of signals in each component
def calculate_component_angles(s, T, theta, MixFirstAngle, MixLastAngle, numItr):
    # Calcuate information on signal
    S = np.zeros(s.shape)
    for n in range(s.shape[0]):
        S[n] = np.abs(np.fft.fft(s[n])) / s.shape[1]

    s0_ind = round((2) * T)
    s0_amp = S[0, round((2) * T)]
    s0_rms = np.sqrt(np.mean(s[0] ** 2))
    s1_ind = round(3 * T)
    s1_amp = S[1, round(3 * T)]
    s1_rms = np.sqrt(np.mean(s[1] ** 2))
    s2_ind = round(5 * T)
    s2_amp = S[2, round(5 * T)]
    s2_rms = np.sqrt(np.mean(s[2] ** 2))

    # Create empty arrays
    c_angles = np.zeros((2, len(theta), numItr))
    c0_contents = np.zeros((3, len(theta), numItr))
    c1_contents = np.zeros((3, len(theta), numItr))
    # prepare angles to stack so no redoing of cos calc
    m0 = np.array([[math.cos(math.radians(MixFirstAngle))], [math.sin(math.radians(MixFirstAngle))]])
    m1 = np.array([[math.cos(math.radians(MixLastAngle))], [math.sin(math.radians(MixLastAngle))]])

    # Create mixing matrix
    for i in range(len(theta)):  # Create M and calculate x for each theta
        m2 = np.array([[math.cos(math.radians(theta[i]))], [math.sin(math.radians(theta[i]))]])
        # Calcuate M and x
        M = np.hstack((m0, m1, m2))
        x = M @ s  # matrix multiplication of M and s

        for j in range(numItr):  # Do ICA set number of times for each theta
            ica.fit(x.T)  # Create unmixing matric
            U = ica.components_  # Set unmixing matrix to U
            M_ICA = np.linalg.inv(U)  # Take inverse of unmixing matrix to find ICA mixing matrix
            c = U @ x  # Create components matrix
            c0_rms = np.sqrt(np.mean(c[0] ** 2))
            c1_rms = np.sqrt(np.mean(c[1] ** 2))
            m0_ICA = M_ICA[:, 0]
            m1_ICA = M_ICA[:, 1]

            c_angles[0, i, j] = math.degrees(
                math.atan(m0_ICA[1] / m0_ICA[0]))  # m0 is sin, m1 is cos so atan(sin/cos)=atan(tan)=radians
            if c_angles[0, i, j] < -45:
                c_angles[0, i, j] = c_angles[0, i, j] + 180
            c_angles[1, i, j] = math.degrees(math.atan(m1_ICA[1] / m1_ICA[0]))
            if c_angles[1, i, j] < -45:
                c_angles[1, i, j] = c_angles[1, i, j] + 180

            # Calculate DFT of c
            C_fft = np.zeros(c.shape)
            for k in range(c.shape[0]):
                C_fft[k] = np.abs(np.fft.fft(c[k])) / c.shape[1]

            # Use DFT to find amplitude of signals in components
            s0_amp_inC0 = (C_fft[0, s0_ind] / c0_rms) / (s0_amp / s0_rms)
            s1_amp_inC0 = (C_fft[0, s1_ind] / c0_rms) / (s1_amp / s1_rms)
            s2_amp_inC0 = (C_fft[0, s2_ind] / c0_rms) / (s2_amp / s2_rms)
            c0_contents[:, i, j] = np.array([s0_amp_inC0, s1_amp_inC0, s2_amp_inC0])
            if np.max(c0_contents[:, i, j]) > 1:
                c0_contents[:, i, j] = c0_contents[:, i, j] / np.max(c0_contents[:, i, j])

            s0_amp_inC1 = (C_fft[1, s0_ind] / c1_rms) / (s0_amp / s0_rms)
            s1_amp_inC1 = (C_fft[1, s1_ind] / c1_rms) / (s1_amp / s1_rms)
            s2_amp_inC1 = (C_fft[1, s2_ind] / c1_rms) / (s2_amp / s2_rms)
            c1_contents[:, i, j] = np.array([s0_amp_inC1, s1_amp_inC1, s2_amp_inC1])
            if np.max(c1_contents[:, i, j]) > 1:
                c1_contents[:, i, j] = c1_contents[:, i, j] / np.max(c1_contents[:, i, j])

    return c_angles, c0_contents, c1_contents


def graphAmplitudes(x, letter):
    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=[5, 5])
    xymin = min(np.amin(x[0]) - 0.2, np.amin(x[1]) - 0.2)
    xymax = max(np.amax(x[0]) + 0.2, np.amax(x[1]) + 0.2)

    axs.plot(x[0, :], x[1, :], '.k', markersize=0.5)
    axs.set_xlabel('${}_0$ amplitude'.format(letter))
    axs.set_ylabel('${}_1$ amplitude'.format(letter))
    axs.set_xlim(xymin, xymax)
    axs.set_ylim(xymin, xymax)
    axs.grid()


def graphAmplitudeVectors(M, M_ICA):
    # Creates line from (1,-1) to (-1,1) to show reflection
    angle_thresh_x = np.arange(-1, 1, .01)
    angle_thresh_y = -angle_thresh_x
    row_zeros = np.zeros(M.shape[0])
    col_zeros = np.zeros(M.shape[1])

    fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=[5, 5])

    axs.plot(angle_thresh_x, angle_thresh_y, '--k')  # plots dotted line for reflection
    axs.quiver(row_zeros, row_zeros, M_ICA[0], M_ICA[1], angles='xy', scale=np.linalg.norm(M_ICA[:, 0]),
               scale_units='xy', color='blue')  # graphs ICA mixing matrix
    axs.quiver(row_zeros, row_zeros, -M_ICA[0], -M_ICA[1], angles='xy', scale=np.linalg.norm(M_ICA[:, 0]),
               scale_units='xy', color='blue', alpha=0.5)  # reflected mixing matrix
    axs.quiver(col_zeros, col_zeros, M[0], M[1], angles='xy', scale_units='xy', scale=1,
               color='red')  # graphs original signals last so on top
    axs.set_xlabel('1st sensor amplitude')
    axs.set_ylabel('2nd sensor amplitude')
    axs.grid()


def DFTPlot(t, x, N):
    X = np.fft.fft(x)
    freq = np.arange(N) / T
    fig, ax = plt.subplots(3, 1, constrained_layout=True)

    ax[0].set_ylabel("Data")
    ax[0].set_xlabel('Time (s)')
    ax[0].plot(t, x)
    ax[0].set_xlim(0, 1)
    ax[0].grid()

    ax[1].set_ylabel("DFT Amplitude (Real Only)")
    ax[0].set_xlabel('Frequency (Hz)')
    ax[1].plot(freq, np.abs(X) / N)
    ax[1].grid()

    ax[1].set_ylabel("DFT Amplitude")
    ax[0].set_xlabel('Frequency (Hz)')
    ax[2].plot(freq, X.real / N, color=[1, 0, 0])
    ax[2].plot(freq, X.imag / N, color=[0, 0, 1])
    ax[2].legend(('Real', 'Imag'))
    ax[2].grid()


def graph2DFT(T, s, letter):
    S, omega_k = fft_matrix(s, T)  # Create matix of amplitudes for matrix s and freq array
    plt.figure(figsize=[4, 4])  # Set figure size
    constant = 10

    # plot amplitues by frequency for each signal
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude of Fourier coefficient')
    plt.plot(omega_k, S[0], "-sr", markersize=constant)
    plt.plot(omega_k, S[1], "^b", markersize=constant, linestyle='--', dashes=(3, 6))
    plt.legend(('${}_0$'.format(letter), '${}_1$'.format(letter)))
    plt.xlim(-0.2, 6)
    plt.grid()


def graph3DFT(T, s, letter):
    S, omega_k = fft_matrix(s, T)  # Create matix of amplitudes for matrix s and freq array
    plt.figure(figsize=[4, 4])  # Set figure size
    constant = 10

    # Plot amplitues by frequency for each signal
    plt.title('Fourier Transform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude of Fourier coefficient')
    plt.plot(omega_k, S[0], "-sr", markersize=constant)
    plt.plot(omega_k, S[1], "--^b", markersize=constant, linestyle='--', dashes=(3, 6))
    plt.plot(omega_k, S[2], ":*", markersize=constant, color=[0, 1, 0])
    plt.legend(('${}_0$'.format(letter), '${}_1$'.format(letter), '${}_2$'.format(letter)))
    plt.xlim(-0.2, 6)
    plt.grid()


def graph2signals(t, x, letter):
    fig, ax = plt.subplots(2, 1, constrained_layout=True, figsize=[6, 3])

    ymin = min(np.amin(x[0]), np.amin(x[1]))
    ymax = max(np.amax(x[0]), np.amax(x[1]))

    # Graph first signal
    ax[0].set_ylabel('${}_0$'.format(letter))
    ax[0].plot(t, x[0], color=[1, 0, 0])
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(ymin, ymax)
    ax[0].grid()

    # Graph second signal
    ax[1].set_ylabel('${}_1$'.format(letter))
    ax[1].plot(t, x[1], color=[0, 1, 0])
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(ymin, ymax)
    ax[1].grid()

    # X axis label
    ax[0].set_xlabel('Time (s)')


def graph3signals(t, s, letter):
    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=[10, 4])
    ymin = min(np.amin(x[0]), np.amin(x[1]), np.amin(x[2]))
    ymax = max(np.amax(x[0]), np.amax(x[1]), np.amax(x[2]))

    # Title
    ax[0].set_title('Sources')

    # Graph first signal
    ax[0].set_ylabel('${}_0$'.format(letter))
    ax[0].plot(t, s[0], color=[1, 0, 0])
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(ymin, ymax)
    ax[0].grid()

    # Graph second signal
    ax[1].set_ylabel('${}_1$'.format(letter))
    ax[1].plot(t, s[1], color=[0, 1, 0])
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(ymin, ymax)
    ax[1].grid()

    # Graph third signal
    ax[2].set_ylabel('${}_2$'.format(letter))
    ax[2].plot(t, s[2], color=[0, 0, 1])
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(ymin, ymax)
    ax[2].grid()

    # X axis label
    ax[2].set_xlabel('Time (s)')


def sourceComponentAngleGraph(MixFirstAngle, MixLastAngle, theta, c_angles, c0_contents, c1_contents):
    plt.figure(figsize=[5, 5])  # Set figure size

    plt.plot(0, MixFirstAngle, 'o', markersize=12, color=[1, 0, 0])
    plt.plot(0, MixLastAngle, 'o', markersize=12, color=[0, 1, 0])
    plt.plot(0, theta[0], 'o', markersize=12, color=[0, 0, 1])

    plt.plot(1, c_angles[0], 'o', markersize=12, color=c0_contents)
    plt.plot(1, c_angles[1], 'o', markersize=12, color=c1_contents)

    plt.xticks([0, 1], ['Source angle', 'Component angle'])
    plt.xlim(-0.5, 1.5)
    plt.ylabel('Angle')
    plt.grid()


def plot_component_mixes(c_angles, c0_contents, c1_contents, MixFirstAngle, MixLastAngle, theta, numItr):
    plt.xlabel(r'$\theta$ for third source')
    plt.ylabel('Angles in degrees')
    plt.axhline(y=MixFirstAngle, color=[1, 0, 0], linestyle='-')
    plt.axhline(y=MixLastAngle, color=[0, 1, 0], linestyle='-')
    plt.plot(theta, theta, color=[0, 0, 1])
    plt.grid()

    for i in range(len(theta)):
        for j in range(numItr):
            plt.plot(theta[i], c_angles[0, i, j], 'o', markersize=12, color=c0_contents[:, i, j], alpha=1 / numItr)
            plt.plot(theta[i], c_angles[1, i, j], 'o', markersize=12, color=c1_contents[:, i, j], alpha=1 / numItr)
