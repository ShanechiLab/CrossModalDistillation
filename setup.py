from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="cross_modal_distillation",
    packages=find_packages(),
    install_requires=requirements,
)
