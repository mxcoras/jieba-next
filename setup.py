import os

from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "jieba_next.jieba_next_functions",
        ["src/jieba_next/jieba_next_functions.pyx"],
        extra_compile_args=["/O2" if os.name == "nt" else "-O3"],
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
