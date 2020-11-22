"""Core functions to call to calculate the entrogram/entropy values."""

import numpy as np
from . import classifier
from . import tools


def global_entropy(Classifier, base=np.e):
    """Calculate global entropy of some data.

    From an :obj:`entrogrammer.classifier.BaseClassifier`, calculate the
    global entropy of the classified data.

    Parameters
    ----------
    Classifier: :obj:`entrogrammer.classifier.BaseClassifier`
        Any initialized class from `classifier.py` that has had the
        `classify()` method run.

    base: int, float, optional
        Logarithmic base for the entropy calculation. Same as the
        `scipy.stats.entropy()` base parameter meaning it takes a default
        value of `e` (natural logarithm) if not specified.

    Returns
    -------
    HG: float
        The global entropy of the classified data array

    """
    # type check the classifier
    if isinstance(Classifier, classifier.BaseClassifier) is False:
        raise TypeError('Classifier must be a BaseClassifier, '
                        'was: %s', str(type(Classifier)))
    elif Classifier.classified is None:
        raise ValueError('`Classifier.classify()` method must be run first.')

    # type check base
    if ((int(base) == 2) or (int(base) == 10) or (base == np.e)) is False:
        raise TypeError('base, if specified, must be valid log base, '
                        'was: %s', str(type(base)))

    # calculate the global entropy
    HG = tools.calculate_HG(Classifier.classified, base)

    return HG


def local_entropy(Classifier, scale, base=np.e):
    """Calculate local entropy of some data at a particular scale.

    From an :obj:`entrogrammer.classifier.BaseClassifier`, calculate the
    local entropy of the classified data at a particular scale.

    Parameters
    ----------
    Classifier: :obj:`entrogrammer.classifier.BaseClassifier`
        Any initialized class from `classifier.py` that has had the
        `classify()` method run.

    scale: int, tuple
        Scale or window size over which to compute the local entropy.
        This is provided as a float, or a tuple.
        If the length of the tuple is less than the number of dimensions in
        the data, the last value in the tuple will be applied to the
        dimensions unaccounted for. Conversely, if the length of the tuple is
        greater than the number of dimensions in the data, N, only the first
        N values of the tuple will be used.

    base: int, float, optional
        Logarithmic base for the entropy calculation. Same as the
        `scipy.stats.entropy()` base parameter meaning it takes a default
        value of `e` (natural logarithm) if not specified.

    Returns
    -------
    HL: float
        The averaged local entropy of the classified data array for the
        specified scale

    """
    # type check the classifier
    if isinstance(Classifier, classifier.BaseClassifier) is False:
        raise TypeError('Classifier must be a BaseClassifier, '
                        'was: %s', str(type(Classifier)))
    elif Classifier.classified is None:
        raise ValueError('`Classifier.classify()` method must be run first.')

    # type check base
    if ((int(base) == 2) or (int(base) == 10) or (base == np.e)) is False:
        raise TypeError('base, if specified, must be valid log base, '
                        'was: %s', str(type(base)))

    # type check the scale
    if type(scale) == int:
        win_size = scale
    elif type(scale) == tuple:
        dims = np.shape(Classifier.classified)
        # 1-D case
        if len(dims) == 1:
            win_size = [0]  # init win_size variable
            if scale[0] == int:
                win_size[0] = scale[0]  # if integer assignment is simple
            else:
                # if not try to assign from tuple
                try:
                    win_size[0] = int(scale[0])
                except Exception:
                    raise TypeError('value in position 0 of scale was not '
                                    ' an `int` / could not be made an `int`.')
        # other dimensions not yet supported
        else:
            raise NotImplementedError('Only 1-D data currently supported.')
    else:
        raise TypeError('scale must be an `int` or `tuple`, '
                        'was: %s', str(type(scale)))

    # calculate local entropy
    HL = tools.calculate_HL(Classifier.classified, win_size, base)

    return HL
