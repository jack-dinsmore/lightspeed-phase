import numpy as np
import re, os
from scipy.interpolate import interp1d
import pint
from pint.models import model_builder
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
pint.logging.setup(level="WARNING")

OBSERVATORY = "Las Campanas Observatory"
STORAGE_INTERVAL = 10*60 # Minutes of phases to store
pint.observatory.topo_obs.TopoObs(OBSERVATORY, location=EarthLocation.of_site(OBSERVATORY))

def from_dms(s):
    # Convert the passed string to hms from degrees
    d, m, s = s.split(':')
    x = np.abs(float(d)) + float(m) / 60 + float(s)/3600
    return np.sign(float(d)) * x

def from_hms(s):
    # Convert the passed string from hms to degrees
    h, m, s = s.split(':')
    return (float(h) + float(m) / 60 + float(s)/3600) * 360 / 24

def get_all_ephemerides_name(obs_name):
    """Get all ephemerides available and put the obe that closest matches obs_name at the top"""
    split_name = re.split(" |_|,|\.|-", obs_name)
    if not os.path.exists("ephemerides"):
        raise Exception("You have no ephemerides stored (store them in the ephemerides directory)")
    
    filenames = []
    scores = []
    for filename in os.listdir("ephemerides"):
        score = 0
        for chunk in split_name:
            if chunk.lower() in filename.lower():
                score += len(chunk)
        scores.append(score)
        filenames.append(filename)

    if len(scores) == 0:
        raise Exception("You have no ephemerides stored (store them in the ephemerides directory)")
    best_filename = filenames[np.argmax(scores)]
    other_filenames = [f for f in filenames if f != best_filename]

    return [best_filename] + list(np.sort(other_filenames))

def get_all_ephemerides_ra_dec(tel_ra, tel_dec):
    """Get all ephemerides available and put the one that closest matches the provided ra, dec at the top"""
    if not os.path.exists("ephemerides"):
        raise Exception("You have no ephemerides stored (store them in the ephemerides directory)")
    ra = from_hms(tel_ra)
    dec = from_dms(tel_dec)
    filenames = []
    distances = []
    stretch = np.cos(dec * np.pi/180)
    for filename in os.listdir("ephemerides"):
        ephem_ra = None
        ephem_dec = None
        with open(f"ephemerides/{filename}") as f:
            for line in f.readlines():
                if line.startswith("RAJ"):
                    ephem_ra = from_hms(line.split()[1])
                if line.startswith("DECJ"):
                    ephem_dec = from_dms(line.split()[1])
        
        filenames.append(filename)
        if ephem_ra is None or ephem_dec is None:
            distances.append(np.inf)
        else:
            distances.append((ephem_ra - ra)**2 * stretch**2 + (ephem_dec - dec)**2)

    if len(distances) == 0:
        raise Exception("You have no ephemerides stored (store them in the ephemerides directory)")
    best_filename = filenames[np.argmin(distances)]
    other_filenames = [f for f in filenames if f != best_filename]

    return [best_filename] + list(np.sort(other_filenames))
    

class Ephemeris:
    def __init__(self, filename, gps_time):
        self.model = model_builder.get_model(f"ephemerides/{filename}")
        self.nu = self.model["F0"].value
        self.ephem = self.model["EPHEM"].value
        self.gps_time = gps_time
        self.storage_index = 0
        self.load_new_phases()
    
    def load_new_phases(self):
        self.timestamps = np.arange(STORAGE_INTERVAL) + self.storage_index * STORAGE_INTERVAL
        times_mjd = self.timestamps / (3600 * 24) + self.gps_time
        toas = pint.toa.get_TOAs_array(
            Time(times_mjd, format="mjd", scale="utc"),
            freqs=np.ones(len(times_mjd)) * 500 * u.THz,  # Dummy frequency
            errors=np.ones(len(times_mjd)) * 1 * u.us,  # Dummy errors
            ephem=self.ephem,
            obs=OBSERVATORY,
        )
        phases = self.model.phase(toas)
        delta_phase_int = phases.int - np.min(phases.int)
        self.phases = delta_phase_int.astype(np.float64) + phases.frac.astype(np.float64)
        self.interpolator = interp1d(self.timestamps, self.phases, bounds_error=False, fill_value="extrapolate") # Extrapolate. This will extrapolate the calculated phases throughout the last frame, which is out of bounds of the interpolator.
        self.storage_index += 1

    def set_freq(self, freq):
        self.model["F0"].value = freq
        self.recalc()

    def recalc(self):
        self.storage_index -= 1
        self.load_new_phases()

    def get_phase(self, timestamp):
        
        """
        Get the phase of an event with the provided ephemeris and timestamp
        """
        if timestamp > self.timestamps[-1]:
            print("Calculating new phases with PINT")
            self.load_new_phases()
        phase = (self.interpolator)(timestamp)
        phase -= np.floor(phase)
        return phase