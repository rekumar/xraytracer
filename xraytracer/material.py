import os
import json
import numpy as np


packageDir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(packageDir, "include", "xrfEmissionLines.json"), "r") as f:
    emissionLines = json.load(f)


def scattering_factor(element, energy):
    """
    returns the real and imaginary x ray scattering factors for an element at a given energy (keV).
    """

    dataDir = os.path.join(packageDir, "include", "scatfacts")

    fid = os.path.join(dataDir, "{0}.nff".format(str.lower(element)))
    with open(fid, "r") as f:
        data = np.genfromtxt(f)[1:, :]
    e_raw = data[:, 0] / 1000  # convert eV to keV
    f1_raw = data[:, 1]
    f2_raw = data[:, 2]

    f1 = np.interp(energy, e_raw, f1_raw)
    f2 = np.interp(energy, e_raw, f2_raw)
    return f1, f2


class Material:
    """
    Class that, for a defined material, can generate x-ray attenuation coefficients
    and related values.
    """

    def __init__(self, elements, density):
        """
        elements: dictionary of elements and their molar fraction of the material. 
                            ie, for FeCu2: {'Fe':1, 'Cu':2} 
        density: overall density of material (g/cm3)
        """
        self.elements = elements
        self.density = density

    def attenuation_coefficient(self, energy):
        """
        returns x-ray attenuation coefficient, in cm-1, given:
            energy: x-ray energy(ies) (keV)
        """
        Na = 6.022e23
        c = (1e-19) / (np.pi * 0.9111)  # keV*cm^2
        energy = np.array(energy)
        f2 = 0
        mass = 0
        for i, (el, num) in enumerate(self.elements.items()):
            _, f2_ = scattering_factor(el, energy)
            f2 += num * f2_
            mass += num * molar_mass(el)

        mu = (self.density * Na / mass) * (2 * c / energy) * f2
        return mu

    def attenuation_length(self, energy):
        """
        returns x-ray attenuation length (distance for transmitted intensity to
        decay to 1/e), in cm
        """
        mu = self.attenuation_coefficient(energy)
        return 1 / mu

    def transmission(self, thickness, energy):
        """
        returns fraction of x-ray intensity transmitted through a sample, defined by
            thickness: path length of x rays (cm)
            energy: x-ray energy (keV)
        """

        mu = self.attenuation_coefficient(energy)
        t = np.exp(-mu * thickness)

        return t

    def phase_delay(self, thickness, energy):
        """
        calculates phase delay (radians) of photons passing through material slab.
        useful for predicting phase contrast in ptychography measurements.
        """
        r_e = 2.8179403227e-13
        # classical radius of electron, cm
        h = 4.135667516e-18
        # plancks constant, keV/sec
        c = 299792458e2
        # speed of light, cm/s
        Na = 6.022e23  # avogadros number, atoms/mol

        wl = h * c / np.asarray(energy)  # photon wavelengths, cm

        f1 = 0
        mass = 0
        for el, num in self.elements.items():
            f1_, _ = scattering_factor(el, energy)
            f1 += num * f1_
            mass += num * molar_mass(el)

        delta = f1 * (self.density * Na / mass) * (r_e / 2 / np.pi) * (wl ** 2)
        phase_delay = 2 * np.pi * delta * thickness / wl

        return phase_delay
