from __future__ import division, print_function, absolute_import

import warnings
from . import _minpack

import numpy as np
from numpy import (atleast_1d, dot, take, triu, shape, eye,
                    transpose, zeros, product, greater, array,
                    all, where, isscalar, asarray, inf, abs,
                    finfo, inexact, issubdtype, dtype)
from .optimize import OptimizeResult, _check_unknown_options

error = _minpack.error

__all__ = ['fsolve', 'leastsq', 'fixed_point', 'curve_fit']


def curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, **kw):
    """
    Use non-linear least squares to fit a function, f, to data.
    Assumes ``ydata = f(xdata, *params) + eps``
    Parameters
    ----------
    f : callable
    The model function, f(x, ...). It must take the independent
    variable as the first argument and the parameters to fit as
    separate remaining arguments.
    xdata : An M-length sequence or an (k,M)-shaped array
    for functions with k predictors.
    The independent variable where the data is measured.
    ydata : M-length sequence
    The dependent data --- nominally f(xdata, ...)
    p0 : None, scalar, or N-length sequence
    Initial guess for the parameters. If None, then the initial
    values will all be 1 (if the number of parameters for the function
    can be determined using introspection, otherwise a ValueError
    is raised).
    sigma : None or M-length sequence, optional
    If not None, these values are used as weights in the
    least-squares problem.
    absolute_sigma : bool, optional
    If False, `sigma` denotes relative weights of the data points.
    The returned covariance matrix `pcov` is based on *estimated*
    errors in the data, and is not affected by the overall
    magnitude of the values in `sigma`. Only the relative
    magnitudes of the `sigma` values matter.
    If True, `sigma` describes one standard deviation errors of
    the input data points. The estimated covariance in `pcov` is
    based on these values.
    Returns
    -------
    popt : array
    Optimal values for the parameters so that the sum of the squared error
    of ``f(xdata, *popt) - ydata`` is minimized
    pcov : 2d array
    The estimated covariance of popt. The diagonals provide the variance
    of the parameter estimate. To compute one standard deviation errors
    on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
    How the `sigma` parameter affects the estimated covariance
    depends on `absolute_sigma` argument, as described above.
    See Also
    --------
    leastsq
    Notes
    -----
    The algorithm uses the Levenberg-Marquardt algorithm through `leastsq`.
    Additional keyword arguments are passed directly to that algorithm.
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.optimize import curve_fit
    >>> def func(x, a, b, c):
    ... return a * np.exp(-b * x) + c
    >>> xdata = np.linspace(0, 4, 50)
    >>> y = func(xdata, 2.5, 1.3, 0.5)
    >>> ydata = y + 0.2 * np.random.normal(size=len(xdata))
    >>> popt, pcov = curve_fit(func, xdata, ydata)
    """
    if p0 is None:
        # determine number of parameters by inspecting the function
        import inspect
        args, varargs, varkw, defaults = inspect.getargspec(f)
        if len(args) < 2:
            msg = "Unable to determine number of fit parameters."
            raise ValueError(msg)
        if 'self' in args:
            p0 = [1.0] * (len(args)-2)
        else:
            p0 = [1.0] * (len(args)-1)

    # Check input arguments
    if isscalar(p0):
        p0 = array([p0])

    ydata = np.asanyarray(ydata)
    if isinstance(xdata, (list, tuple)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        xdata = np.asarray(xdata)

    args = (xdata, ydata, f)
    if sigma is None:
        func = _general_function
    else:
        func = _weighted_general_function
        args += (1.0 / asarray(sigma),)

    # Remove full_output from kw, otherwise we're passing it in twice.
    return_full = kw.pop('full_output', False)
    res = leastsq(func, p0, args=args, full_output=1, **kw)
    (popt, pcov, infodict, errmsg, ier) = res

    if ier not in [1, 2, 3, 4]:
        msg = "Optimal parameters not found: " + errmsg
        raise RuntimeError(msg)

    if pcov is None:
        # indeterminate covariance
        pcov = zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
    elif not absolute_sigma:
        if len(ydata) > len(p0):
            s_sq = (asarray(func(popt, *args))**2).sum() / (len(ydata) - len(p0))
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov

