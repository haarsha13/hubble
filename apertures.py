from dLux.optical_systems import OpticalSystem
import jax.numpy as np

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx

from abcdLux.lct import *
from abcdLux.abcd import *

"""
Aperture and optical models
"""


class HSTMainAperture(dl.CompoundAperture):
    """
    HST OTA aperture, including spiders and mirror pads
    """
    softening : float
    def __init__(self, transformation=dl.CoordTransform(rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "mirror" : dl.CircularAperture(
                radius = 1.2,
                softening=self.softening,
            ),
            "spider" : dl.Spider(
                width = 0.022*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening=self.softening,
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.330*1.2,
                occulting = True,
                softening = self.softening
            ),
            "pad_1" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (0.8921*1.2, 0),
                ),
                softening = self.softening
            ),
            "pad_2" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4615*1.2, 0.7555*1.2),
                ),
                softening = self.softening
            ),
            "pad_3" : dl.CircularAperture(
                radius = 0.065*1.2,
                occulting = True,
                transformation=dl.CoordTransform(
                    translation = (-0.4564*1.2, -0.7606*1.2),
                ),
                softening=self.softening
            )
        }



class NICMOSColdMask(dl.CompoundAperture):
    """
    NIC1 cold mask
    """
    softening : float
    def __init__(self, transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.04)),rotation=np.pi/4), softening=0.25):
        self.normalise = True
        self.transformation = transformation
        self.softening = softening
        self.apertures = {
            "outer" : dl.CircularAperture(
                radius = 1.2*0.955,
                softening = self.softening,
                #normalise=True
            ),
            "spider" : dl.Spider(
                width = 0.077*1.2,
                angles = np.asarray([0, 90, 180, 270]),
                softening = self.softening
            ),
            "secondary" : dl.CircularAperture(
                radius = 0.372*1.2,
                occulting = True,
                softening = self.softening
            ),
        }



class NICMOSOptics(dl.AngularOpticalSystem):
    """
    Optical system for in-focus NICMOS optics
    """
    def __init__(self, wf_npixels, psf_npixels, oversample, psf_oversample=1, n_zernikes = 26):
        super().__init__(
            wf_npixels,
            2.4,
            [
                dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4), softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
                dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(4,4+n_zernikes),
                    coefficients = np.zeros(n_zernikes)
                ),
            ],
            psf_npixels,
            0.0431/psf_oversample,
            oversample
        )


class NICMOSFresnelOptics(dl.AngularOpticalSystem):
    """
    Optical system for NICMOS optics with Fresnel defocus
    """
    defocus: np.ndarray
    fnumber: np.ndarray
    def __init__(self, wf_npixels, psf_npixels, oversample, defocus, fnumber, n_zernikes = 26):
        self.diameter=2.4
        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = 0.0432
        self.oversample = oversample
        self.defocus = defocus
        self.fnumber = fnumber

        layers = []

        layers += [
            dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
        ]

        layers += [dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )]

        self.layers = dlu.list2dictionary(layers, ordered=True)
    
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):

        wf = dl.Wavefront(wavelength, self.wf_npixels, diameter=self.diameter)
        wf = wf.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wf = layer(wf)

        u_in = wf.phasor

        fl = self.fnumber*self.diameter
        abcd = compose_abcd([abcd_lens(fl), abcd_free_space(fl + self.defocus)])

        N_in = self.wf_npixels
        dx_in = self.diameter/self.wf_npixels

        N_out = self.psf_npixels*self.oversample
        dx_out = 40e-6/self.oversample

        # patch over abcdLux bug
        x_in = dlu.nd_coords(N_in, dx_in)
        x_out = dlu.nd_coords(N_out, dx_out)

        u_out = lct_prop(u_in, x_in, x_out, wavelength, abcd)

        wf = dl.Wavefront(wavelength, N_out, diameter=N_out*dx_out).set("phasor", u_out)

        if return_wf:
            return wf
        return wf.psf

def abcd_magnification(m):
    return np.array([[m, 0.], [0., 1/m]])

class NICMOSSecondaryFresnelOptics(dl.AngularOpticalSystem):
    """
    Optical system for NICMOS optics with Fresnel defocus expressed as secondary mirror despace
    """
    defocus: np.ndarray
    despace: np.ndarray
    mag: np.ndarray
    def __init__(self, wf_npixels, psf_npixels, oversample, defocus, despace, mag, n_zernikes = 26):
        self.diameter=2.4
        self.wf_npixels = wf_npixels
        self.psf_npixels = psf_npixels
        self.psf_pixel_scale = 0.0432
        self.oversample = oversample
        self.defocus = defocus
        self.despace = despace
        self.mag = mag

        layers = []

        layers += [
            dl.CompoundAperture([
                    ("main_aperture",HSTMainAperture(transformation=dl.CoordTransform(rotation=np.pi/4),softening=2)),
                    ("cold_mask",NICMOSColdMask(transformation=dl.CoordTransform(translation=np.asarray((-0.05,-0.05)),rotation=np.pi/4, compression=np.asarray([1.,1.])), softening=2)),
                ],normalise=True, transformation=dl.CoordTransform(rotation=0)),
        ]

        layers += [dl.AberratedAperture(
                    dl.layers.CircularAperture(1.2, transformation=dl.CoordTransform()),
                    noll_inds=np.arange(5,5+n_zernikes),
                    coefficients = np.zeros(n_zernikes),
                )]

        self.layers = dlu.list2dictionary(layers, ordered=True)
    
    def propagate_mono(self, wavelength, offset=np.zeros(2), return_wf=False):

        wf = dl.Wavefront(wavelength, self.wf_npixels, diameter=self.diameter)
        wf = wf.tilt(offset)

        # Apply layers
        for layer in list(self.layers.values()):
            wf = layer(wf)

        u_in = wf.phasor

        abcd = compose_abcd([
            abcd_lens(5.52085),
            abcd_free_space(4.907028205 + self.despace),
            abcd_lens(-0.6790325),
            abcd_free_space(6.3919974 + self.despace + self.defocus),
            abcd_magnification(self.mag),
        ])

        N_in = self.wf_npixels
        dx_in = self.diameter/self.wf_npixels

        N_out = self.psf_npixels*self.oversample
        dx_out = 40e-6/self.oversample

        # patch over abcdLux bug
        x_in = dlu.nd_coords(N_in, dx_in)
        x_out = dlu.nd_coords(N_out, dx_out)

        u_out = lct_prop(u_in, x_in, x_out, wavelength, abcd)

        wf = dl.Wavefront(wavelength, N_out, diameter=N_out*dx_out).set("phasor", u_out)

        if return_wf:
            return wf
        return wf.psf
