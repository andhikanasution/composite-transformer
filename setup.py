from setuptools import setup, find_packages

setup(
    name="composite_stress_prediction",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
