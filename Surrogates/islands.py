import numba
import numpy as np

numba.jit()


def evaluate_islands_on_set(parameter_combinations):
    y = np.zeros(parameter_combinations.shape[0])
    num_params = parameter_combinations.shape[1]

    if num_params == 1:
        for i, rho in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)
    elif num_params == 2:
        for i, (rho, alpha) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 3:
        for i, (rho, alpha, phi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 4:
        for i, (rho, alpha, phi, pi) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             pi=pi, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    elif num_params == 5:
        for i, (rho, alpha, phi, pi, eps) in enumerate(parameter_combinations):
            gdp = island_abm(rho=rho, alpha=alpha, phi=phi,
                             pi=pi, eps=eps, _RNG_SEED=0)
            y[i] = calibration_measure(gdp)

    return y


numba.jit()


def calibration_measure(log_GDP):
    """ Calibration Measure
    Input
    -----
    GDP : array, required, length = [,T]
    Output
    ------
    agdp : float
    Average GDP growth rate
    """
    T = log_GDP.shape[0]
    log_GDP = log_GDP[~np.isinf(log_GDP)]
    log_GDP = log_GDP[~np.isnan(log_GDP)]
    if log_GDP.shape[0] > 0:
        GDP_growth_rate = (log_GDP[-1] - log_GDP[0]) / T
    else:
        GDP_growth_rate = 0

    return GDP_growth_rate


numba.jit()


def calibration_condition(average_GDP_growth_rate, calibration_thresh=1.0):
    return average_GDP_growth_rate >= calibration_thresh


numba.jit()


def island_abm(rho=0.01,
               alpha=1.5,
               phi=0.4,
               pi=0.4,
               eps=0.1,
               lambda_param=1,
               T=100,
               N=50,
               _RNG_SEED=0):
    """ Islands growth model
    Parameters
    ----------
    rho :
    alpha :
    phi : float, required
    eps :
    lambda_param: (Default = 1)
    T : int, required
    The number of periods for the simulation
    N : int, optional (Default = 50)
    Number of firms
    _RNG_SEED : int, optional (Default = 0)
    Random number seen
    Output
    ------
    GDP : array, length = [,T]
    Simulated GPD
    """
    # Set random number seed
    np.random.seed(_RNG_SEED)

    T_2 = int(T / 2)

    GDP = np.zeros((T, 1))

    # Distributions
    # Precompute random binomial draws
    xy = np.random.binomial(1, pi, (T, T))
    xy[T_2, T_2] = 1

    # Containers
    s = np.zeros((T, T))
    A = np.ones((N, 6))

    # Initializations
    A[:, 1] = T_2
    A[:, 2] = T_2
    m = np.zeros((T, T))
    m[T_2, T_2] = N
    dest = np.zeros((N, 2))

    """ Begin ABM Code """
    for t in range(T):
        w = np.zeros((N, N))
        signal = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    if A[j, 0] == 1:
                        w[i, j] = np.exp(-rho * (np.abs(A[j, 1] - A[i, 1]) + \
                                                 np.abs(A[j, 2] - A[i, 2])))

                        if np.random.rand() < w[i, j]:
                            signal[i, j] = s[int(A[j, 1]), int(A[j, 2])]

            if A[i, 0] == 1:
                A[i, 4] = s[int(A[i, 1]), int(A[i, 2])] * \
                          m[int(A[i, 1]), int(A[i, 2])] ** alpha
                A[i, 3] = s[int(A[i, 1]), int(A[i, 2])]

            if A[i, 0] == 3:
                A[i, 4] = 0
                rnd = np.random.rand()
                if rnd <= 0.25:
                    A[i, 1] += 1
                else:
                    if rnd <= 0.5:
                        A[i, 1] -= 1
                    else:
                        if rnd <= 0.75:
                            A[i, 2] += 1
                        else:
                            A[i, 2] -= 1

                if xy[int(A[i, 1]), int(A[i, 2])] == 1:
                    A[i, 0] = 1
                    m[int(A[i, 1]), int(A[i, 2])] += 1
                    if m[int(A[i, 1]), int(A[i, 2])] == 1:
                        s[int(A[i, 1]), int(A[i, 2])] = \
                            (1 + int(np.random.poisson(lambda_param))) * \
                            (A[i, 1] + A[i, 2]) + phi * A[i, 5] + np.random.randn()

            if (A[i, 0] == 1) and (np.random.rand() <= eps):
                A[i, 0] = 3
                A[i, 5] = A[i, 4]
                m[int(A[i, 1]), int(A[i, 2])] -= 1

            if t > T / 100:
                if A[i, 0] == 2:
                    A[i, 4] = 0
                    if dest[i, 0] != A[i, 1]:
                        if dest[i, 0] > A[i, 1]:
                            A[i, 1] += 1
                        else:
                            A[i, 1] -= 1
                    else:
                        if dest[i, 1] != A[i, 2]:
                            if dest[i, 1] > A[i, 2]:
                                A[i, 2] += 1
                            else:
                                A[i, 2] -= 1
                    if (dest[i, 0] == A[i, 1]) and (dest[i, 1] == A[i, 2]):
                        A[i, 0] = 1
                        m[int(dest[i, 0]), int(dest[i, 1])] += 1
                if A[i, 0] == 1:
                    best_sig = np.max(signal[i, :])
                    if best_sig > s[int(A[i, 1]), int(A[i, 2])]:
                        A[i, 0] = 2
                        A[i, 5] = A[i, 4]
                        m[int(A[i, 1]), int(A[i, 2])] -= 1
                        index = np.where(signal[i, :] == best_sig)[0]
                        if index.shape[0] > 1:
                            ind = int(index[int(np.random.uniform(0, len(index)))])
                        else:
                            ind = int(index)
                        dest[i, 0] = A[ind, 1]
                        dest[i, 1] = A[ind, 2]

        GDP[t, 0] = np.sum(A[:, 4])

    #JH fix around the divide by zero error supressed in original code
    np.seterr(divide='ignore')
    log_GDP = np.log(GDP)
    np.seterr(divide='warn')
    log_GDP[np.isneginf(log_GDP)] = 0

    return log_GDP
