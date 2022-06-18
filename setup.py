import os

from setuptools import setup, Extension


def get_extension():
    file_path = os.path.abspath(__file__)
    base, _ = os.path.split(file_path)
    pybind_path = os.path.join(base, "dependence/pybind11/include")
    eigen_path = os.path.join(base, "dependence/eigen")
    core_path = os.path.join(base, "pr/prase_core.cpp")
    prase_core_module = Extension(name='prase_core', sources=[core_path], include_dirs=[pybind_path, eigen_path],
                                  extra_compile_args=["-std=c++14"])
    return [prase_core_module]


setup(name="prase",
      version="1.0.0",
      packages=["prase", "pr", "se", "utils", "sb"],
      author="qizhyuan",
      author_email="qizhyuan@gmail.com",
      ext_modules=get_extension())
