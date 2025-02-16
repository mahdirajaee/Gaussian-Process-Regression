import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(Np=200, Nm=21, T=10):
    """
    Generates a synthetic Gaussian random process by passing white Gaussian noise through a Gaussian filter.
    """
    Nprev = 2 * Nm
    t_h = np.arange(-Nprev, Nprev)
    h = np.exp(-(t_h / T) ** 2)  # Gaussian impulse response
    h = h / np.linalg.norm(h)
    
    x = np.random.randn(Np)  # White Gaussian process
    y = np.convolve(x, h, mode='same')  # Filtered Gaussian process
    t = np.arange(len(y))
    
    return t, y, t_h, h

def plot_impulse_response(t_h, h):
    """Plots the impulse response of the Gaussian filter."""
    plt.figure()
    plt.plot(t_h, h)
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('h(t)')
    plt.title('Impulse Response')
    plt.show()

def plot_realization(t, y):
    """Plots a realization of the Gaussian random process."""
    plt.figure()
    plt.plot(t, y)
    plt.grid()
    plt.xlabel('t (s)')
    plt.ylabel('y(t)')
    plt.title('Realization of the Gaussian Random Process')
    plt.show()

def main():
    t, y, t_h, h = generate_synthetic_data()
    plot_impulse_response(t_h, h)
    plot_realization(t, y)
    
if __name__ == "__main__":
    main()