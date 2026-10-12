import jax.numpy as np
import jax.random as jr
import jax.scipy as jsp
from jax import Array
import jax.tree_util as jtu
from jax.flatten_util import ravel_pytree
import jax

import dLux as dl
import dLux.utils as dlu

import zodiax as zdx
import equinox as eqx

from abc import abstractmethod

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D


from apertures import *
from detectors import *
from spectra import *
from filters import *
from stats import gauss_log_likelihood

"""
Models
"""

class Exposure(zdx.Base):
    filename: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    mjd: Array
    exptime: Array
    wcs: object = eqx.field(static=True)
    pam: Array
    data: Array
    err: Array
    bad: Array


    fit: object

    def __init__(self, filename, name, filter, data, err, bad, fit, mjd, exptime, wcs, pam):
        """
        Initialise exposure
        """
        self.filename = filename
        self.target = name
        self.filter = filter
        self.data = data
        self.err = err
        self.bad = bad

        self.mjd = mjd

        self.fit = fit
        self.exptime = exptime
        self.wcs = wcs
        self.pam = pam
    
    def get_key(self, param):
        return self.fit.get_key(self, param)

    def map_param(self, param):
        return self.fit.map_param(self, param)
    
    @property
    def key(self):
        return self.filename

class BlankExposure(Exposure):
    """
    Dummy exposure for injection-recovery
    """
    def __init__(self, name, filter, fit):
        self.filter = filter
        self.filename = f"{name}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        self.data = 0.
        self.err = 0.
        self.bad = 0.
        self.exptime = 0.
        self.pam = 0.


class InjectedExposure(Exposure):
    """
    Injected exposure - generates data from the model and fit and spoofs other parameters
    """
    def __init__(self, name, filter, fit, model, t_exp, n_exp, read_noise=10.):
        self.filter = filter
        self.filename = f"{name}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        gain = 3

        generated_data = self.fit(model, self) * t_exp * gain

        err = np.sqrt(generated_data/(gain*t_exp) + read_noise**2)/np.sqrt(n_exp)

        data = jr.normal(jr.key(0),generated_data.shape)*err + generated_data

        #err = np.sqrt(data/(gain*exptime) + err**2)

        self.data = data/t_exp/gain
        self.err = err/t_exp/gain
        self.bad = np.zeros(self.data.shape)

        self.exptime = t_exp
        self.pam = 0.

class LoadedExposure(Exposure):
    """
    Builds exposure using arbitrary provided data
    """
    def __init__(self, name, filter, fit, data, err, bad):
        self.filter = filter
        self.filename = f"{name}"
        self.target = name
        self.fit = fit
        self.mjd = 0.0
        self.wcs = None

        self.data = data
        self.err = err
        self.bad = bad
        self.exptime = 0.
        self.pam = 0.

def exposure_from_file(fname, fit, extra_bad=None, crop=None):
    """
    builds exposure from provided file and fit object
    """

    hdr = fits.getheader(fname, ext=0)
    image_hdr = fits.getheader(fname, ext=1)

    # load data etc from file
    data = fits.getdata(fname, ext=1)
    err = fits.getdata(fname, ext=2)
    info = fits.getdata(fname, ext=3)

    # mask out bad pixels which appear along the detector chip boundary
    detector_mask = np.full((256, 256), False, dtype=bool).at[127:130, :].set(True)

    bad = np.asarray((err==0.0) | (info&256) | (info&64) | (info&32) | detector_mask)
    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    # retrieve metadata
    wcs = WCS(image_hdr)
    pam = hdr['NPFOCUSP']
    filename = hdr['ROOTNAME']
    name = hdr['TARGNAME']
    filter = hdr['FILTER']
    exptime = float(hdr['EXPTIME'])
    gain = float(hdr['ADCGAIN'])
    mjd = hdr['EXPSTART']

    # print pipeline version - pipelines after 4.2 don't include photon noise
    print(hdr["CAL_VER"])

    if crop:
        w = WCS(image_hdr)

        y, x = numpy.unravel_index( numpy.nanargmax(data), data.shape )

        print(x, y)

        centre = SkyCoord( w.pixel_to_world(x, y), unit="deg" )

        data_cutout = Cutout2D( data, centre, crop, wcs=w, mode="partial", fill_value=np.nan )

        err_cutout = Cutout2D( err, centre, crop, wcs=w, mode="partial", fill_value=np.nan )
        bad_cutout = Cutout2D( bad, centre, crop, wcs=w, mode="partial", fill_value=True )

        data = np.asarray(data_cutout.data)
        err = np.asarray(err_cutout.data)
        bad = np.asarray(bad_cutout.data)

    # add in extra bad pixels
    if extra_bad is not None:
        bad = bad | extra_bad

    # nan out bad pixels in data and error
    err = np.where(bad, np.nan, np.asarray(err, dtype=float))
    data = np.where(bad, np.nan, np.asarray(data, dtype=float))

    # add poisson noise to error
    err_with_poisson = np.sqrt(data/(gain*exptime) + err**2)
    bad_with_poisson = np.isnan(err_with_poisson)

    return Exposure(filename, name, filter, data, err_with_poisson, bad_with_poisson, fit, mjd, exptime, wcs, pam)

class ModelFit(zdx.Base):
    """
    Model fitting object.  Controls how parameters are (potentially hierarchically) 
    applied to the optical model and calculates the log-likelihood.  
    """

    source: object

    @abstractmethod
    def update_source(self, model, exposure):
        """
        Update source parameters
        """
        pass

    def get_key(self, exposure, param):
        """
        determines key for lookup in the model parameters.  allows for hierarchical
        parameter estimation - some parameters can be shared between exposures while
        others can be fitted independently.  

        possible keys include:
            exposure.key - a unique key for each exposure, permitting independent fitting
            "global" - shared between all exposures (usually something that should be refactored)
            str(round(exposure.mjd)) - all exposures on same day share the same key
            exposure.target - shared for all observations of the same target
            exposure.filter - shared for all observations of the same filter
        """
        match param:            
            case "aberrations":
                #return "global"
                return exposure.key
            case "breathing":
                return exposure.key
            case "cold_mask_shift":
                return "global"#exposure.key#"global"
                #return str(round(exposure.mjd))
            case "cold_mask_rot":
                return "global"
            case "cold_mask_scale":
                return "global"
            case "cold_mask_shear":
                return "global"
            case "primary_rot":
                return "global"
            case "primary_scale":
                return "global"
            case "primary_shear":
                return "global"
            case "primary_distortion" | "cold_mask_distortion":
                return "global"
            case "defocus":
                return exposure.key
            case "bias":
                return exposure.key
            case "jitter":
                return exposure.key
            case "despace":
                return exposure.key
            case "quadrature":
                return exposure.key
            case _: raise ValueError(f"Parameter {param} has no key")
    
    def map_param(self, exposure, param):
        """
        determines full location of model parameter. global parameters are simply "param"
        while other parameters are "param.key"
        """
        if param in ["aberrations", "cold_mask_shift", "cold_mask_rot", "cold_mask_scale", "cold_mask_shear", "primary_rot", "primary_scale", "primary_shear", "bias", "jitter", "primary_distortion", "cold_mask_distortion", "defocus", "despace", "quadrature"]:
            return f"{param}.{exposure.get_key(param)}"
        return param
    
    def update_optics(self, model, exposure):
        """
        Applies parameters to the optical model, scaling as necessary
        """

        optics = model.optics
        if "aberrations" in model.params.keys():
            coefficients = model.get(self.map_param(exposure, "aberrations"))*1e-9
            optics = optics.set("AberratedAperture.coefficients", coefficients)
        
        if "cold_mask_shift" in model.params.keys():
            translation = model.get(self.map_param(exposure, "cold_mask_shift"))*1e-2
            optics = optics.set("cold_mask.transformation.translation", translation)

        if "cold_mask_scale" in model.params.keys():
            compression = model.get(self.map_param(exposure, "cold_mask_scale"))
            optics = optics.set("cold_mask.transformation.compression", compression)

        if "cold_mask_rot" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "cold_mask_rot")))
            optics = optics.set("cold_mask.transformation.rotation", rotation)

        if "cold_mask_shear" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "cold_mask_shear")))
            optics = optics.set("cold_mask.transformation.shear", rotation)
        
        if "outer_radius" in model.params.keys():
            radius = model.get(self.map_param(exposure, "outer_radius"))
            optics = optics.set("cold_mask.outer.radius", radius)
        
        if "secondary_radius" in model.params.keys():
            radius = model.get(self.map_param(exposure, "secondary_radius"))
            optics = optics.set("cold_mask.secondary.radius", radius)
        
        if "spider_width" in model.params.keys():
            radius = model.get(self.map_param(exposure, "spider_width"))
            optics = optics.set("cold_mask.spider.width", radius)

        if "primary_scale" in model.params.keys():
            compression = model.get(self.map_param(exposure, "primary_scale"))
            optics = optics.set("main_aperture.transformation.compression", compression)
            optics = optics.set("AberratedAperture.aperture.transformation.compression", compression)

        if "primary_rot" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "primary_rot")))
            optics = optics.set("main_aperture.transformation.rotation", rotation)
            optics = optics.set("AberratedAperture.aperture.transformation.rotation", rotation)

        if "primary_shear" in model.params.keys():
            rotation = dlu.deg2rad(model.get(self.map_param(exposure, "primary_shear")))
            optics = optics.set("main_aperture.transformation.shear", rotation)
            optics = optics.set("AberratedAperture.aperture.transformation.shear", rotation)
        
        if "rot" in model.params.keys():
            rot = dlu.deg2rad(model.get(self.map_param(exposure, "rot")))
            optics = optics.set("CompoundAperture.transformation.rotation", rot)
        if "scale" in model.params.keys():
            scale = model.get(self.map_param(exposure, "scale"))
            optics = optics.set("psf_pixel_scale", scale)
        if "softening" in model.params.keys():
            softening = model.get(self.map_param(exposure, "softening"))
            optics = optics.set("main_aperture.softening", softening)
            optics = optics.set("cold_mask.softening", softening)
            optics = optics.set("AberratedAperture.aperture.softness", softening)
        if "displacement" in model.params.keys():
            disp = model.get(self.map_param(exposure, "displacement"))
            optics = optics.set("displacement", disp)

        if "defocus" in model.params.keys():
            disp = model.get(self.map_param(exposure, "defocus"))
            optics = optics.set("defocus", disp*1e-2)
        
        if "despace" in model.params.keys():
            disp = model.get(self.map_param(exposure, "despace"))
            optics = optics.set("despace", disp*1e-6)
        
        if "fnumber" in model.params.keys():
            fn = model.get(self.map_param(exposure, "fnumber"))
            optics = optics.set("fnumber", fn)

        if "mag" in model.params.keys():
            fn = model.get(self.map_param(exposure, "mag"))
            optics = optics.set("mag", fn)

        if "primary_distortion" in model.params.keys():
            dist = model.get(self.map_param(exposure, "primary_distortion"))
            optics = optics.set("main_aperture.transformation.distortion", dist)

        if "cold_mask_distortion" in model.params.keys():
            dist = model.get(self.map_param(exposure, "cold_mask_distortion"))
            optics = optics.set("cold_mask.transformation.distortion", dist)        

        return optics

    def update_detector(self, model, exposure):
        """
        applies relevant parameters to the detector model
        """
        detector = model.detector

        if "bias" in model.params.keys():
            bias = model.get(self.map_param(exposure, "bias"))
            detector = detector.set("bias.value", bias)
        if "jitter" in model.params.keys():
            jitter = model.get(self.map_param(exposure, "jitter"))
            detector = detector.set("jitter.sigma", np.abs(jitter))
        return detector

    def __call__(self, model, exposure):
        """
        Forward models an exposure from the provided model
        """
        source = self.update_source(model, exposure)
        optics = self.update_optics(model, exposure)
        detector = self.update_detector(model, exposure)

        psfs = optics.model(source, return_psf=True)
        psf = psfs.data.sum(tuple(range(psfs.ndim)))
        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)
    
    def loglike(self, model, exposure, per_pix=False, return_im=False):
        """
        computes the log-likelihood of an exposure given the specified model
        """
        psf = self(model, exposure)

        data = exposure.data
        err = exposure.err
        bad = exposure.bad
        err = np.where(bad, 1., err)

        # add excess noise multiplicatively
        if "quadrature" in model.params.keys():
            quad_error = 10**model.get(self.map_param(exposure, "quadrature"))
            err = err*quad_error

        posterior_im = gauss_log_likelihood(psf, (data, err, bad))
        if return_im:
            return posterior_im
        
        if per_pix:
            return np.nanmean(posterior_im)
        return np.nansum(posterior_im)
        
# base ModelFit only handles the instrument state, its subclasses are responsible for the source parameters        

class SinglePointFit(ModelFit):
    """
    Model for a single point source, potentially with a spectrum or time-varying intensity
    """
    time_series: bool = eqx.field(static=True)

    def __init__(self, spectrum_basis, filter, time_series=False, precombined=False, wavels=None):
        nwavels, nbasis = spectrum_basis.shape
        if precombined:
            self.source = dl.PointSource(spectrum=PreCombinedBasisSpectrum(wavels, np.zeros(nbasis), spectrum_basis))
        else:
            wv, inten = calc_throughput(filter, nwavels)
            self.source = dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))
        self.time_series=time_series
    
    def get_key(self, exposure, param):
        """
        source-specific keys
        """
        if param == "positions":
            return exposure.key
        elif param == "spectrum" or param == "flux":
            if self.time_series:
                return exposure.key
            else:    
                return f"{exposure.target}_{exposure.filter}"
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        """
        source-specific parameter mapping
        """
        if param in ["positions", "spectrum"]:
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def update_source(self, model, exposure):
        """
        Apply model to the source
        """
        
        spectrum_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))

        source = self.source.set("spectrum.basis_weights", spectrum_coeffs)
        source = source.set("flux", source.spectrum.flux)
        source = source.set("position", model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432))
        
        return source    



class BreathingFit(ModelFit):
    """
    Model for long exposure with linear breathing
    """
    ns: int = eqx.field(static=True)
    def __init__(self, ns):
        self.source = dl.PointSource(wavelengths=[1])
        self.ns = ns

    def get_key(self, exposure, param):
        if param == "breathing":
            return exposure.key
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param == "breathing":
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def __call__(self, model, exposure):
        source = self.update_source(model, exposure)
        detector = self.update_detector(model, exposure)

        breathing = model.get(exposure.fit.map_param(exposure, "breathing"))
        aberrations = model.get(exposure.fit.map_param(exposure, "aberrations"))

        defocuses = np.linspace(-breathing, breathing, self.ns)

        psf = 0.0

        for i in range(self.ns):
            ab = aberrations.at[0].add(defocuses[i])
            model = model.set(exposure.fit.map_param(exposure, "aberrations"), ab)
            optics = self.update_optics(model, exposure)
            psfs = optics.model(source, return_psf=True)
            psf = psf + psfs.data.sum(tuple(range(psfs.ndim)))/self.ns

        pixel_scale = psfs.pixel_scale.mean()

        psf_obj = dl.PSF(psf, pixel_scale)
        
        return detector.model(psf_obj, return_psf=False)
    
class BreathingSinglePointFit(SinglePointFit, BreathingFit):
    """
    Single point fit but with breathing
    """
    def __init__(self, spectrum, nwavels, ns):
        SinglePointFit.__init__(self, spectrum, nwavels)
        BreathingFit.__init__(self, ns)


class BinaryFit(ModelFit):
    """
    Binary fit, allowing each component to have distinct spectra
    """
    def __init__(self, spectrum_basis, filter):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)
        self.source = dl.Scene([
            ("primary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))), 
            ("secondary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis)))
        ])
            
    def get_key(self, exposure, param):
        if param == "positions":
            return exposure.key
        elif param == "primary_spectrum" or param == "secondary_spectrum":
            return f"{exposure.target}_{exposure.filter}"
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param in ["positions", "primary_spectrum", "secondary_spectrum"]:
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def update_source(self, model, exposure):
        """
        Apply model to the source.  Note this largely replicates the machinery of a
        dLux BinarySource, but is more general in providing arbitrary spectra
        """
        primary_coeffs = model.get(exposure.fit.map_param(exposure, "primary_spectrum"))
        secondary_coeffs = model.get(exposure.fit.map_param(exposure, "secondary_spectrum"))

        source = self.source.set("primary.spectrum.basis_weights", primary_coeffs)
        source = source.set("primary.flux", source.primary.spectrum.flux)
        source = source.set("secondary.spectrum.basis_weights", secondary_coeffs)
        source = source.set("secondary.flux", source.secondary.spectrum.flux)


        position = model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432)
        separation = model.get(exposure.fit.map_param(exposure, "separation"))*dlu.arcsec2rad(0.0432)
        position_angle = dlu.deg2rad(model.get(exposure.fit.map_param(exposure, "position_angle")))


        positions = dlu.positions_from_sep(position, separation, position_angle)

        source = source.set("primary.position", positions[0])
        source = source.set("secondary.position", positions[1])
        
        return source

class PointSourceContrastFit(ModelFit):
    """
    Basically another binary source with a different parameterisation
    """
    def __init__(self, spectrum_basis, filter):
        nwavels, nbasis = spectrum_basis.shape
        wv, inten = calc_throughput(filter, nwavels)
        self.source = dl.Scene([
            ("primary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis))), 
            ("secondary",dl.PointSource(spectrum=CombinedBasisSpectrum(wv, inten, np.zeros(nbasis), spectrum_basis)))
        ])
            
    def get_key(self, exposure, param):
        if param == "positions":
            return exposure.key
        elif param == "spectrum" or param == "secondary_spectrum" or param == "secondary_position":
            return f"{exposure.target}_{exposure.filter}"
        else:
            return super().get_key(exposure, param)
    
    def map_param(self, exposure, param):
        if param in ["positions", "spectrum", "secondary_spectrum", "secondary_position"]:
            return f"{param}.{exposure.get_key(param)}"
        else:
            return super().map_param(exposure, param)

    def update_source(self, model, exposure):
        primary_coeffs = model.get(exposure.fit.map_param(exposure, "spectrum"))
        secondary_coeffs = model.get(exposure.fit.map_param(exposure, "secondary_spectrum"))

        source = self.source.set("primary.spectrum.basis_weights", primary_coeffs)
        source = source.set("primary.flux", source.primary.spectrum.flux)
        source = source.set("secondary.spectrum.basis_weights", secondary_coeffs)
        source = source.set("secondary.flux", source.secondary.spectrum.flux)


        position = model.get(exposure.fit.map_param(exposure, "positions"))*dlu.arcsec2rad(0.0432)

        secondary_position = model.get(exposure.fit.map_param(exposure, "secondary_position"))*dlu.arcsec2rad(0.0432)

        source = source.set("primary.position", position)

        source = source.set("secondary.position", secondary_position)
        
        return source


class BaseModeller(zdx.Base):
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def __getitem__(self, key):

        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]

        return values

class NICMOSModel(BaseModeller):
    """
    Wrapper that holds all the relevant objects for the NICMOS model

    maybe not necessary anymore?
    """

    filters: dict
    optics: NICMOSOptics
    detector: NICMOSDetector

    def __init__(self, exposures, params, optics, detector):
        self.optics = optics
        self.detector = detector
        self.params = params
        self.filters = {}

        for filter in [e.filter for e in exposures]:
            #print(filter)
            spec = filter_files[filter]
            spec = spec.at[:,0].divide(1e10)
            self.filters[filter] = spec[::5,:]    





class ModelParams(BaseModeller):
    """
    Class to hold the model parameters dictionary
    """
    def __getitem__(self, key):
        return self.params[key]

    def __getattr__(self, key):

        # Make the object act like a real dictionary
        if hasattr(self.params, key):
            return getattr(self.params, key)

        if key in self.params.keys():
            return self.params[key]

        for sub_key, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)

        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set("params", dict([(param, getattr(values, param)) for param in self.keys()]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys()]))

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree.map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree.map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def map(self, fn):
        return jax.tree.map(lambda x: fn(x), self)

    # Re-name this donate, and it counterpart accept, receive?
    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(list(self.keys()), list(self.values()))

    def partition(self, params):
        """params can be a model params object or a list of keys"""
        if isinstance(params, ModelParams):
            params = list(params.params.keys())
        return (
            ModelParams({param: self[param] for param in params}),
            ModelParams({param: self[param] for param in self.keys() if param not in params}),
        )

    def combine(self, params2):
        return ModelParams({**self.params, **params2.params})

    def jacfwd(self, fn, n_batch=1):
        X, unravel_fn = ravel_pytree(self)
        Xs = np.array_split(X, n_batch)
        rebuild = lambda X_batch, index: X.at[index : index + len(X_batch)].set(X_batch)
        lens = np.cumsum(np.array([len(x) for x in Xs]))[:-1]
        starts = np.concatenate([np.array([0]), lens])

        @eqx.filter_jacfwd
        def batched_jac_fn(x, index):
            model_params = unravel_fn(rebuild(x, index))
            return eqx.filter_jit(fn)(model_params)

        return np.concatenate([batched_jac_fn(x, index) for x, index in zip(Xs, starts)], axis=-1)
