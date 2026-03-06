from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ROOT = Path(__file__).resolve().parent

ext_modules = [
    Pybind11Extension(
        "gooomoku_cpp",
        [str(ROOT / "cpp" / "gooomoku_cpp.cpp")],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native", "-DNDEBUG"],
    )
]

setup(
    name="gooomoku-cpp",
    version="0.1.0",
    description="C++ backend for gooomoku",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
