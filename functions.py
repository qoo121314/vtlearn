"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(*, function, name, arity, wrap=True):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    wrap : bool, optional (default=True)
        When running in parallel, pickling of custom functions is not supported
        by Python's default pickler. This option will wrap the function using
        cloudpickle allowing you to pickle your solution, but the evolution may
        run slightly more slowly. If you are running single-threaded in an
        interactive Python session or have no need to save the model, set to
        `False` for faster runs.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if function.__code__.co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity, function.__code__.co_argcount))
    if not isinstance(name, str):
        raise ValueError('name must be a string, got %s' % type(name))
    if not isinstance(wrap, bool):
        raise ValueError('wrap must be an bool, got %s' % type(wrap))

    # Check output shape
    args = [np.ones((10,1)) for _ in range(arity)]
    try:
        function(*args)
    except (ValueError, TypeError):
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,1):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros((10,3)) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones((10,3)) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    if len(function(*args).shape) < 2:
        raise ValueError('supplied function return %s does not return the shape '
                         '(n, m) array.' % function(*args).shape)
    
    if wrap:
        return _Function(function=wrap_non_picklable_objects(function),
                         name=name,
                         arity=arity)
    return _Function(function=function,
                     name=name,
                     arity=arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def _sigmoid(x1):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        return 1 / (1 + np.exp(-x1))


add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}




def _pow_3(x): 
    return x**3

pow_3 = make_function(function=_pow_3,name='pow3',arity=1)


def __crossed_function_(data, function='mean'):
    if function == 'mean':
        return np.array(pd.DataFrame(data).mean(axis=1))
    elif function == 'rank':
        return np.array(pd.DataFrame(data).rank(axis=1))
    elif function == 'max':
        return np.array(pd.DataFrame(data).max(axis=1))
    elif function == 'min':
        return np.array(pd.DataFrame(data).min(axis=1))

def _cross_rank(x): 
    return __crossed_function_(x, 'rank')
cross_rank = make_function(function=_cross_rank,name='cross_rank',arity=1)

def __ts_function_(data, window, function ='delay'):
    if function == 'delay':
        return np.array(pd.DataFrame(data).shift(window).fillna(method='bfill'))
    elif function == 'delta':
        return np.array(pd.DataFrame(data).diff(window).fillna(0))

def _delay_1(x): 
    return __ts_function_(x, 1, 'delay')
delay_1 = make_function(function=_delay_1,name='delay_1',arity=1)

def _delay_2(x): 
    return __ts_function_(x, 2, 'delay')
delay_2 = make_function(function=_delay_2,name='delay_2',arity=1)

def _delay_3(x): 
    return __ts_function_(x, 3, 'delay')
delay_3 = make_function(function=_delay_3,name='delay_3',arity=1)

def _delta_1(x): 
    return __ts_function_(x, 1, 'delta')
delta_1 = make_function(function=_delta_1,name='delta_1',arity=1)

def _delta_2(x): 
    return __ts_function_(x, 2, 'delta')
delta_2 = make_function(function=_delta_2,name='delta_2',arity=1)

def _delta_3(x): 
    return __ts_function_(x, 3, 'delta')
delta_3 = make_function(function=_delta_3,name='delta_3',arity=1)


def __rolling_ts_function_(data, window, function='mean'):
    if function == 'mean':
        return np.array(pd.DataFrame(data).rolling(window, min_periods=1).mean())
    elif function == 'max':
        return np.array(pd.DataFrame(data).rolling(window, min_periods=1).max())
    elif function == 'min':
        return np.array(pd.DataFrame(data).rolling(window, min_periods=1).min())
    elif function == 'rank':
        return np.array(pd.DataFrame(data).rolling(window, min_periods=1).rank())
    elif function == 'corr':
        return np.array(pd.DataFrame(data[0]).rolling(window, min_periods=1).corr(pd.DataFrame(data[1])).fillna(0))

def _rolling_mean_5(x): 
    return __rolling_ts_function_(x, 5, 'mean')
ts_mean_5 = make_function(function=_rolling_mean_5,name='ts_mean_5',arity=1)

def _rolling_mean_10(x): 
    return __rolling_ts_function_(x, 10, 'mean')
ts_mean_10 = make_function(function=_rolling_mean_10,name='ts_mean_10',arity=1)

def _rolling_mean_15(x): 
    return __rolling_ts_function_(x, 15, 'mean')
ts_mean_15 = make_function(function=_rolling_mean_15,name='ts_mean_15',arity=1)

def _rolling_mean_20(x): 
    return __rolling_ts_function_(x, 20, 'mean')
ts_mean_20 = make_function(function=_rolling_mean_20,name='ts_mean_20',arity=1)


def _rolling_max_5(x): 
    return __rolling_ts_function_(x, 5, 'max')
ts_max_5 = make_function(function=_rolling_max_5,name='ts_max_5',arity=1)

def _rolling_max_10(x): 
    return __rolling_ts_function_(x, 10, 'max')
ts_max_10 = make_function(function=_rolling_max_10,name='ts_max_10',arity=1)


def _rolling_max_15(x): 
    return __rolling_ts_function_(x, 15, 'max')
ts_max_15 = make_function(function=_rolling_max_15,name='ts_max_15',arity=1)


def _rolling_max_20(x): 
    return __rolling_ts_function_(x, 20, 'max')
ts_max_20 = make_function(function=_rolling_max_20,name='ts_max_20',arity=1)


def _rolling_min_5(x): 
    return __rolling_ts_function_(x, 5, 'min')
ts_min_5 = make_function(function=_rolling_min_5,name='ts_min_5',arity=1)

def _rolling_min_10(x): 
    return __rolling_ts_function_(x, 10, 'min')
ts_min_10 = make_function(function=_rolling_min_10,name='ts_min_10',arity=1)

def _rolling_min_15(x): 
    return __rolling_ts_function_(x, 15, 'min')
ts_min_15 = make_function(function=_rolling_min_15,name='ts_min_15',arity=1)

def _rolling_min_20(x): 
    return __rolling_ts_function_(x, 20, 'min')
ts_min_20 = make_function(function=_rolling_min_20,name='ts_min_20',arity=1)


def _rolling_rank_5(x): 
    return __rolling_ts_function_(x, 5, 'rank')
ts_rank_5 = make_function(function=_rolling_rank_5,name='ts_rank_5',arity=1)

def _rolling_rank_10(x): 
    return __rolling_ts_function_(x, 10, 'min')
ts_rank_10 = make_function(function=_rolling_rank_10,name='ts_rank_10',arity=1)

def _rolling_rank_15(x): 
    return __rolling_ts_function_(x, 15, 'min')
ts_rank_15 = make_function(function=_rolling_rank_15,name='ts_rank_15',arity=1)

def _rolling_rank_20(x): 
    return __rolling_ts_function_(x, 20, 'min')
ts_rank_20 = make_function(function=_rolling_rank_20,name='ts_rank_20',arity=1)


def _rolling_corr_5(x, y): 
    return __rolling_ts_function_((x,y), 5, 'corr')
ts_corr_5 = make_function(function=_rolling_corr_5,name='ts_corr_5',arity=2)

def _rolling_corr_10(x, y): 
    return __rolling_ts_function_((x,y), 10, 'corr')
ts_corr_10 = make_function(function=_rolling_corr_10,name='ts_corr_10',arity=2)

def _rolling_corr_15(x, y): 
    return __rolling_ts_function_((x,y), 15, 'corr')
ts_corr_15 = make_function(function=_rolling_corr_15,name='ts_corr_15',arity=2)

def _rolling_corr_20(x, y): 
    return __rolling_ts_function_((x,y), 20, 'corr')
ts_corr_20 = make_function(function=_rolling_corr_20,name='ts_corr_20',arity=2)





_extended_function_map = {'add': add2,
                        'sub': sub2,
                        'mul': mul2,
                        'div': div2,
                        'sqrt': sqrt1,
                        'log': log1,
                        'abs': abs1,
                        'neg': neg1,
                        'inv': inv1,
                        'max': max2,
                        'min': min2,
                        'sin': sin1,
                        'cos': cos1,
                        'tan': tan1,
                        'ts_mean_5':ts_mean_5, 
                        'ts_mean_10':ts_mean_10, 
                        'ts_mean_15':ts_mean_15,
                        'ts_mean_20':ts_mean_20,
                        'ts_max_5':ts_max_5, 
                        'ts_max_10':ts_max_10, 
                        'ts_max_15':ts_max_15, 
                        'ts_max_20':ts_max_20,
                        'ts_min_5':ts_min_5, 
                        'ts_min_10':ts_min_10, 
                        'ts_min_15':ts_min_15, 
                        'ts_min_20':ts_min_20,
                        'ts_rank_5':ts_rank_5, 
                        'ts_rank_10':ts_rank_10, 
                        'ts_rank_15':ts_rank_15, 
                        'ts_rank_20':ts_rank_20,
                        'ts_corr_5':ts_corr_5, 
                        'ts_corr_10':ts_corr_10, 
                        'ts_corr_15':ts_corr_15, 
                        'ts_corr_20':ts_corr_20,
                        'cross_rank':cross_rank,
                        'delta_1':delta_1,
                        'delta_2':delta_2,
                        'delta_3':delta_3,
                        'delay_1':delay_1,
                        'delay_2':delay_2,
                        'delay_3':delay_3
                        }