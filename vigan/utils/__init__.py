__all__ = ['convert', 'ds9', 'fit', 'imutils', 'prof', 'sendmail']

from . import convert
from . import fit
from . import imutils
from . import prof

from .sendmail import sendmail
from .convert import *

# ds9 module not supported in Windows
try:
    from . import ds9
except ImportError:
    print('Warning: vigan.utils.ds9 is not available in Windows')
