import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Force Matplotlib to use TkAgg (works better on macOS)
import matplotlib.pyplot as plt

def generate_synthetic_data(Np=200, Nm=21, T=10):
    """ Generate a synthetic Gaussian random process """
    Nprev = 2 * Nm
    t_h = np.arange(-Nprev, Nprev)  # Time axis
    h = np.exp(- (t_h / T) ** 2)  # Gaussian filter impulse response
    h /= np.linalg.norm(h)  # Normalize filter

    x = np.random.randn(Np)  # White Gaussian noise
    y = np.convolve(x, h, mode='same')  # Filtered Gaussian process
    t = np.arange(len(y))  # Time axis

    return t, y, h, t_h

def plot_synthetic_data(t, y, h, t_h):
    """ Plot impulse response and realization of the Gaussian process """
    plt.figure()
    plt.plot(t_h, h)
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('h(t)')
    plt.title('Impulse Response')
    plt.show()  # Ensure figure is displayed

    plt.figure()
    plt.plot(t, y)
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('y(t)')
    plt.title('Realization of the Gaussian Random Process')
    plt.show()  # Ensure figure is displayed

if __name__ == "__main__":
    t, y, h, t_h = generate_synthetic_data()
    plot_synthetic_data(t, y, h, t_h)
