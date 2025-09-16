from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            target="jieba_next.jieba_next_rust",
            path="Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    zip_safe=False,
)
