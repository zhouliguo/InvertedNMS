#python build_softnms.py build_ext --inplace

import sys
import numpy as np
#A=sys.path.insert(0, "..")
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
 
ext_module = Extension("soft_nms",["soft_nms.pyx"])
 
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
    include_dirs=[np.get_include()]
)