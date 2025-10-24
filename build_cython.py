import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension


def build(setup_kwargs):
    compile_args = ["-march=native", "-O3", "-msse", "-msse2", "-mfma", "-mfpmath=sse"]
    link_args = []
    include_dirs = [numpy.get_include()]
    libraries = []

    extensions = [
        Extension(
            "*",
            ["**/*.pyx"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            include_dirs=include_dirs,
            libraries=libraries,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    ]

    ext_modules = cythonize(
        extensions, include_path=include_dirs, compiler_directives={"binding": True, "language_level": 3}
    )

    setup_kwargs.update({"ext_modules": ext_modules, "cmdclass": {"build_ext": build_ext}})
