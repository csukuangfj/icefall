# https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/vits/monotonic_align/setup.py
"""Setup cython code."""

try:
    from Cython.Build import cythonize
except ModuleNotFoundError as ex:
    raise RuntimeError(f'{ex}\nPlease run:\n pip install cython')
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    """Overwrite build_ext."""

    def finalize_options(self):
        """Prevent numpy from thinking it is still in its setup process."""
        _build_ext.finalize_options(self)
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


exts = [
    Extension(
        name="core",
        sources=["core.pyx"],
    )
]
setup(
    name="monotonic_align",
    ext_modules=cythonize(exts, language_level=3),
    cmdclass={"build_ext": build_ext},
)
