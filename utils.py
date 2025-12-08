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

class ImageCorrector:
    def __init__(self, image_shape, median_cut=3, max_wavelength=2):
        sigma = max_wavelength / 0.05
        width = int(np.ceil(3 * sigma))
        line = np.arange(-width, width+1) / sigma # Units of sigma
        self.kernel = np.sin(2*np.pi*line) / line
        self.kernel[line==0] = 1
        self.kernel /= np.sum(self.kernel)
        self.flat = convolve(np.ones(image_shape[1]), self.kernel, mode="same")
        self.median_cut = median_cut
        self.max_wavelength = max_wavelength

    def correct(self, image):
        # Median cut
        image_median = np.median(image)
        image[np.abs(image - image_median) > self.median_cut] = image_median

        # High pass filter
        median_profile = np.median(image, axis=0)
        median_profile -= convolve(median_profile, self.kernel, mode="same") / self.flat

        return image - median_profile

class RollingBuffer:
    def __init__(self, limit, image_size):
        self.image_size = image_size
        self.data = np.zeros((limit, *image_size))
        self.data_index = 0
        self.data_valid = np.zeros(limit, bool) # Start with every data set not being valid

        # The images are combined into "chunks" to reduce size in memory. A chunk is 1/50 of the full buffer.
        self.chunk_limit = limit//50 + 1 # The number of images to stack into a chunk before adding the chunk to the buffer
        self.current_chunk = np.zeros((self.chunk_limit, *image_size))
        self.chunk_index = 0
        # No need to create a chunk_valid variable. Everything after the chunk index is invalid.

    def push(self, image):
        if len(self.current_chunk) <= 1:
            self.data[self.data_index] = image
            self.data_valid[self.data_index] = True
            self.data_index += 1
        else:
            # Add the image to the chunk
            self.current_chunk[self.chunk_index] = image
            self.chunk_index += 1
            if self.chunk_index >= len(self.current_chunk):
                # If the chunk is done, add it to the data
                self.chunk_index = 0
                self.data[self.data_index] = np.sum(self.current_chunk, axis=0)
                self.data_valid[self.data_index] = True
                self.data_index += 1

        # Loop the data index
        self.data_index = self.data_index % len(self.data)

    def num(self):
        return self.chunk_limit*np.sum(self.data_valid) + self.chunk_index

    def get(self):
        output = np.sum(self.data[self.data_valid], axis=0)
        if len(self.current_chunk) > 1:
            output += np.sum(self.current_chunk[:self.chunk_index], axis=0)
        return output.astype(float) / max(1, self.num())
    
    def clear(self):
        self.data_index = 0
        self.data_valid &= False # Set all data to invalid

    def extend(self, new_limit):
        if new_limit == len(self.data):
            # The new limit is the same as the old limit. Do nothing
            return
        
        # Commit the current chunk to data
        self.data[self.data_index] += np.sum(self.current_chunk, axis=0)
        self.data_index += 1
        self.data_index = self.data_index % len(self.data)

        # Extend the current chunk array
        chunk_limit = new_limit//50 + 1 # The number of images to stack into a chunk before adding the chunk to the buffer
        self.current_chunk = np.zeros((chunk_limit, *self.image_size))
        self.chunk_index = 0

        # Create the new data storage
        new_data = np.zeros((new_limit, *self.image_size))
        new_data_valid = np.zeros(new_limit, bool)
        new_data_index = 0
        
        # Iterate through all the currently stored data and save all the valid options
        for _ in range(len(self.data)):
            if self.data_valid[self.data_index]:
                new_data[new_data_index] = self.data[self.data_index]
                new_data_valid[new_data_index] = True
                new_data_index += 1
                new_data_index = new_data_index % len(new_data)
            self.data_index += 1
            self.data_index = self.data_index % len(self.data)
        
        # Overwrite the old data
        self.data_index = new_data_index
        self.data = new_data
        self.data_valid = new_data_valid