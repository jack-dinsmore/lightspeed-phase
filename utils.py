import numpy as np
from scipy.signal import convolve
        
def check_phase(phase, phase_range):
    """
    Returns a bool indicating whether phase is inside phase_range.
    """
    if phase_range[0] < phase_range[1]:
        return (phase_range[0] < phase) and (phase < phase_range[1])
    else:
        return (phase_range[0] < phase) or (phase < phase_range[1])
    
def get_phase_duration(phase_range):
    """
    Returns the length of the phase bin
    """
    if phase_range[0] < phase_range[1]:
        return phase_range[1] - phase_range[0]
    else:
        return 1 + phase_range[1] - phase_range[0]
    
class ImageBuf:
    def __init__(self, image_shape):
        self.image = np.zeros(image_shape)
        self.n = 0

    def push(self, image):
        self.image += image
        self.n += 1

    def get(self):
        return self.image.astype(float) / self.n

    def num(self):
        return self.n

    def clear(self):
        self.image *= 0
        self.n = 0

class LightCurveBuf:
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.fluxes = np.zeros(n_bins)
        self.exposures = np.zeros(n_bins)


    def push(self, flux, phase):
        index = int(phase * self.n_bins)
        self.fluxes[index] += flux
        self.exposures[index] += 1

    def get(self):
        output = self.fluxes.astype(float) / self.exposures
        output[self.exposures==0] = 0
        return output

    def num(self):
        return np.sum(self.exposures)

    def clear(self):
        self.exposures *= 0
        self.fluxes *= 0
