from xraytracer.material import Material
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from scipy.ndimage import center_of_mass


def self_absorption_film(
    material: Material, thickness, incidentenergy, xrfenergy, sampletheta, detectortheta
):
    """
    Calculate x-ray fluorescence self absorption in a planar sample.
    returns fraction of incident beam power that is causes fluorescence, transmits through a sample, and reaches the XRF detector. 
    This calculation assumes no secondary fluorescence/photon recycling. The returned fraction is the apparent signal after 
    incident beam attenuation and exit fluorescence attenuation - dividing the measured XRF value by this fraction should
    approximately correct for self-absorption losses and allow better comparison of fluorescence signals in different energy
    ranges. Reference: https://doi.org/10.1107/S1600577516015721

    Calculations are defined by:

        material: xrf.Material class 
        thickness: Sample thickness - NOT PATH LENGTH (cm)
        incidentenergy: x-ray energy (keV) of incoming beam
        xrfenergy: x-ray energy (keV) of XRF signal
        sampletheta: angle (degrees) between incident beam and sample normal
        detectortheta: angle (degrees) between incident beam and XRF detector axis
    """

    incidentAttCoeff = material.attenuation_coefficient(incidentenergy)
    exitAttCoeff = material.attenuation_coefficient(xrfenergy)

    incident_theta = np.deg2rad(sampletheta)
    exit_theta = np.deg2rad(detectortheta - sampletheta)

    c = np.abs(incidentAttCoeff / np.cos(incident_theta)) + np.abs(
        exitAttCoeff / np.cos(exit_theta)
    )

    xrfFraction = (1 / thickness) * (1 / c) * (1 - np.exp(-c * thickness))

    return xrfFraction


class Particle:
    """
    Calculate sample point-spread function (sPSF) and x-ray fluorescence self absorption factor (SAF) 
    in a sample of arbitrary shape. The sPSF can be used to deconvolute interaction volumes from measured
    data, or to mutually convolve multiple measurements into a single frame of reference. The SAF can be 
    used to correct for self-absorption of fluorecence signal within the sample. Reference: under review
    """

    def __init__(self, z, scale, sample_theta, detector_theta):
        """
        z: 3d array of z values
        scale: centimeters per pixel. can be single value (isotropic scale) or 3-tuple (x, y, z) scales
        sample_theta: angle (degrees) between incident beam and sample normal
        detector_theta: angle (degrees) between incident beam and XRF detector axis
        """
        self.z = z
        self.scale = scale
        self.sample_theta = sample_theta
        self.detector_theta = detector_theta
        self.align_method = "com"
        self._interaction_weight = None

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        scale = np.array(scale)
        if len(scale) == 1:
            scale = np.tile(scale, 3)  # single scale value given, assume same for x,y,z
        self.__scale = scale

    @property
    def sample_theta(self):
        return self.__sample_theta

    @sample_theta.setter
    def sample_theta(self, sample_theta):
        self.__sample_theta = sample_theta
        self.__sample_theta_rad = np.deg2rad(sample_theta)

    @property
    def detector_theta(self):
        return self.__detector_theta

    @detector_theta.setter
    def detector_theta(self, detector_theta):
        self.__detector_theta = detector_theta
        self.__detector_theta_rad = np.deg2rad(detector_theta)

    @property
    def align_method(self):
        return self.__align_method

    @align_method.setter
    def align_method(self, m):
        if m in [0, "leading"]:
            self.__align_method = "leading"
        elif m in [1, "com"]:
            self.__align_method = "com"
        else:
            raise ValueError(".align_method must be 'leading' or 'com'")

    def __check_if_in_simulation_bounds(self, i, j, k):
        for current_position, simulation_bound in zip([i, j, k], self._d.shape):
            if (current_position < 0) or (
                int(round(current_position)) >= simulation_bound
            ):
                return False
        return True

    def raytrace(
        self, step=5, pad_left=None, pad_right=None, chunksize=200, n_workers=4
    ):
        """Trace the path of x-rays through the sample defined by self.z

        :param step: number of mesh points to step each iteration. low=accurate+long, high=coarse+fast, defaults to 5
        :param pad_left: empty pixels to pad to the left. Will autofill to allow beam projection within simulation volume. defaults to None
        :param pad_right: empty pixels to pad to the right. Will autofill to allow beam projection within simulation volume.
        :param chunksize: raytrace uses multiprocessing. This is the chunksize for mp.starmap, defaults to 200
        :param n_workers: number of cores to use during raytracing., defaults to 4
        """
        self.step = step
        self.__step_cm = step * self.scale[2] * 1e-4  # step in units of k (z axis)

        pad_default = (
            np.ceil(
                np.abs(
                    (self.z.max() / np.tan(self.__sample_theta_rad))
                )  # max x displacement based off max z
                / self.scale[1]  # scaled in case x and z scales differ
            ).astype(int)
            + 1
        )  # add 1 to buffer and avoid clipping

        if pad_left is None:
            pad_left = pad_default
        if pad_right is None:
            pad_right = pad_default

        self.__pad_left = pad_left
        self.__pad_right = pad_right
        x_pad = pad_right + pad_left
        numz = int(self.z.max() / self.scale[2])

        self._d = np.full(
            (self.z.shape[0], self.z.shape[1] + x_pad, numz), False
        )  # 3d boolean mask of sample volume - assumes no embedded holes in sample
        for i, j in np.ndindex(*self.z.shape):
            zidx = int(self.z[i, j] / self.scale[2])
            self._d[i, int(j + pad_left), :zidx] = True

        self._incident_steps = [
            [None for j in range(self._d.shape[1])] for i in range(self._d.shape[0])
        ]
        self._emission_steps = [
            [None for j in range(self._d.shape[1])] for i in range(self._d.shape[0])
        ]

        pts = list(np.ndindex(self._d.shape[:2]))
        with mp.Pool(n_workers) as pool:
            raytrace_output = pool.starmap(
                self._trace_incident,
                tqdm.tqdm(pts, total=len(pts)),
                chunksize=chunksize,
            )

        for (i, j), (in_pts, em_steps) in zip(pts, raytrace_output):
            self._incident_steps[i][j] = in_pts
            self._emission_steps[i][j] = em_steps

    def calc_spsf(self, material: Material, incident_energy: float):
        """calculate the sample point-spread function for a given material and incident energy.

        :param material: Material object defined by xraytracer.material.Material
        :param incident_energy: incident photon energy (keV)
        :return: 2D list of dictionaries containining the following keys:
            'coordinate': (x,y,z) mesh coordinate where incident beam intersected sample
            'weight': fraction of total beam attenuation that occured at this point
        """
        self.spsf = []
        self._interaction_weight = np.zeros(self._d.shape[:2])
        step_transmission = material.transmission(self.__step_cm, incident_energy)

        for i, row in enumerate(self._incident_steps):
            row_spsf = []
            for j, ray in enumerate(row):
                this_point = {"coordinate": [], "weight": []}
                incident_power = 1
                for intersection_pt in ray:
                    this_point["coordinate"].append(intersection_pt)
                    this_point["weight"].append(incident_power)
                    incident_power *= step_transmission
                this_point["coordinate"] = np.array(this_point["coordinate"])
                total_weight = np.sum(this_point["weight"])
                this_point["weight"] = np.array(
                    [w / total_weight for w in this_point["weight"]]
                )  # normalize interaction weight so all points sum to 1
                row_spsf.append(this_point)
                self._interaction_weight[i, j] = total_weight
            self.spsf.append(row_spsf)
        self.spsf = self._clip_to_original(np.array(self.spsf), ssf=True)

        return self.spsf

    def convolve_by_spsf(self, source):
        """convolve a source image by the sample point-spread function.

        :param source: 2D numpy array with source image. Should have same dimension at raytracing mesh in self.z
        :return: 2D numpy array with convolved image
        :rtype: [type]
        """
        if source.shape != (len(self.spsf), len(self.spsf[0])):
            raise ValueError("Source image must have same shape as self.spsf")

        output = np.zeros(source.shape)

        #     sourcecp = source.copy()
        #     sourcecp[np.isnan(sourcecp) = 0]
        jmax = source.shape[1] - 1
        for m, mpsf in enumerate(self.spsf):
            for n, k in enumerate(mpsf):
                totweight = 0
                for coord, weight in zip(k["coordinate"], k["weight"]):
                    i, j = coord[0], coord[1]
                    if j > jmax:
                        j = jmax
                        print("offset outside map")
                        output[m, n] += source[i, j] * weight
                        totweight += weight
                output[m, n] /= totweight
        return output

    def calc_self_absorption_factor(
        self, material: Material, incident_energy: float, emission_energy: float
    ):
        """Calculate the self absorption factor for a given material and incident energy.

        :param material: Material object defined by xraytracer.material.Material
        :param incident_energy: incident photon energy (keV)
        :param emission_energy: emission photon energy (keV)
        :return: 2D numpy array with fraction of fluorescence signal remaining after self absorption.

        """
        if self._interaction_weight is None:
            self.calc_ssf(material, incident_energy)

        self.saf = np.zeros(self._d.shape[:2])

        step_transmission_incident = material.transmission(
            self.__step_cm, incident_energy
        )
        step_transmission_emission = material.transmission(
            self.__step_cm, emission_energy
        )

        for i, (incident_row, emission_row) in enumerate(
            zip(self._incident_steps, self._emission_steps)
        ):
            for j, (incident_ray, exit_ray) in enumerate(
                zip(incident_row, emission_row)
            ):
                incident_power = 1
                emission_power = 0
                for intersection_pt, exit_pts in zip(incident_ray, exit_ray):
                    abs_power = incident_power * (
                        1 - step_transmission_incident
                    )  # assume all power attenuated at this step is absorbed + fluoresced
                    emission_power += abs_power * (
                        step_transmission_emission ** exit_pts
                    )  # attenuate fluoresced signal for all intersection steps on the exit path
                    incident_power *= step_transmission_incident  # attenuate incident beam before moving on to next intersection point
                self.saf[i, j] = emission_power
        self.saf = self._clip_to_original(np.array(self.saf))

        return self.saf

    def _clip_to_original(self, x, ssf=False):
        offset_method_lookup = {
            "leading": self._find_offset_leading_edge,
            "com": self._find_offset_com,
        }
        offset_method = offset_method_lookup[self.__align_method]
        alignment_offset = offset_method()

        j_start = self.__pad_left - alignment_offset
        slice_j = slice(j_start, j_start + self.z.shape[1], 1)

        if ssf:
            for m, n in np.ndindex(x.shape):
                if len(x[m, n]["coordinate"]):
                    x[m, n]["coordinate"][:, 1] -= (
                        j_start + 1
                    )  # decrement by one more to account for zero indexing

        return x[:, slice_j]

    def _find_offset_leading_edge(self, threshold=0.1):
        def find_offset_line(i):
            z_line = np.argmin(self._d[i], axis=1)
            z_above_threshold = np.where(z_line > d_thresh)[0]
            if len(z_above_threshold) == 0:
                return np.nan, np.nan
            x_z_min = z_above_threshold.min()
            x_z_max = z_above_threshold.max()

            signal_factor = self._interaction_weight[i]
            signal_above_threshold = np.where(signal_factor > signal_thresh)[0]
            if len(signal_above_threshold) == 0:
                return np.nan, np.nan
            x_signal_min = signal_above_threshold.min()
            x_signal_max = signal_above_threshold.max()

            left_edge_offset = x_z_min - x_signal_min
            right_edge_offset = x_z_max - x_signal_max

            return left_edge_offset, right_edge_offset

        d_thresh = self._d.max() * threshold
        signal_thresh = self._interaction_weight.max() * threshold

        if np.abs(self.sample_theta) <= 90:  # beam enters from right side of sample
            offset = np.nanmean(
                [find_offset_line(i)[1] for i in range(self._d.shape[0])]
            )
        else:
            offset = np.nanmean(
                [find_offset_line(i)[0] for i in range(self._d.shape[0])]
            )

        return int(
            round(offset)
        )  # need to clip the output arrays at integer index values

    def _find_offset_com(self):
        z_com = center_of_mass(self._d.argmin(axis=2))
        signal_com = center_of_mass(self._interaction_weight)
        offset = (
            z_com[1] - signal_com[1]
        )  # only offset in x from beam projection - pencil beam contained in xz plane

        return int(round(offset))

        return offset

    def _trace_emission(self, i, j, k):
        exit_theta = self.__sample_theta_rad - self.__detector_theta_rad
        step_i = 0 * (self.scale[2] / self.scale[1])  # step is in units of k (z axis)
        step_j = (
            self.step * np.cos(exit_theta) * (self.scale[2] / self.scale[0])
        )  # step is in units of k (z axis)
        step_k = self.step * np.sin(exit_theta)
        n_attenuation_steps = 0

        in_bounds = True
        while in_bounds:
            i += step_i
            j += step_j
            k -= step_k

            i_ = int(round(i))
            j_ = int(round(j))
            k_ = int(round(k))
            in_bounds = self.__check_if_in_simulation_bounds(i, j, k)

            if in_bounds and self._d[i_, j_, k_]:  # sample exists at coordinate
                n_attenuation_steps += 1

        return n_attenuation_steps

    def _trace_incident(self, i, j):
        step_i = 0 * (self.scale[2] / self.scale[1])  # step is in units of k (z axis)
        step_j = (
            self.step
            * np.cos(self.__sample_theta_rad)
            * (self.scale[2] / self.scale[0])
        )  # step is in units of k (z axis)
        step_k = self.step * np.sin(self.__sample_theta_rad)

        attenuation_points = []
        n_emission_steps = []

        k = self._d.shape[2] - 1

        in_bounds = True
        while in_bounds:
            i_ = int(round(i))
            j_ = int(round(j))
            k_ = int(round(k))

            if self._d[i_, j_, k_]:  # sample exists at coordinate
                attenuation_points.append((i_, j_, k_))
                n_emission_steps.append(self._trace_emission(i, j, k))

            i -= step_i
            j -= step_j
            k -= step_k
            in_bounds = self.__check_if_in_simulation_bounds(i, j, k)

        return attenuation_points, n_emission_steps
