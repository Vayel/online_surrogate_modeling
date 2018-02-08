import numba
import sobol_seq
import numpy as np

numba.jit


def jsonTransform(parametersRange, samples, extractor):
    parameterNames = parametersRange.keys()
    n_dimensions = len(parameterNames)
    parameterSupports = np.array([i for i in parametersRange.values()])

    outputs = extractor(n_dimensions, samples, parameterSupports)

    parameters = []
    parameterNames = parametersRange.keys()
    for output in outputs:
        parameters.append(dict(zip(parameterNames,output)))

    return parameters

def jsonTransformSobol(parametersRange, samples):
    return jsonTransform(parametersRange,samples,get_sobol_samples)


def jsonTransformOOS(parametersRange, samples):
    return jsonTransform(parametersRange,samples,get_unirand_samples)


def get_sobol_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the Sobol samples
    random_samples = sobol_seq.i4_sobol_generate(n_dimensions, samples)

    # Compute the parameter mappings between the Sobol samples and supports
    sobol_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return sobol_samples


numba.jit()


def get_unirand_samples(n_dimensions, samples, parameter_support):
    """
    """
    # Get the range for the support
    support_range = parameter_support[:, 1] - parameter_support[:, 0]

    # Generate the samples
    random_samples = np.random.rand(n_dimensions, samples).T

    # Compute the parameter mappings between the unirand samples and supports
    unirand_samples = np.vstack([
        np.multiply(s, support_range) + parameter_support[:, 0]
        for s in random_samples])

    return unirand_samples


numba.jit()
