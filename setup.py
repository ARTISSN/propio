# setup.py

from setuptools import setup, find_namespace_packages

# Read requirements from files
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Core requirements
INSTALL_REQUIRES = read_requirements('swapper/requirements.txt')

# Modal-specific requirements
MODAL_REQUIRES = read_requirements('swapper/modal_requirements.txt')

setup(
    name="swapper",
    version="0.1",
    packages=find_namespace_packages(include=["swapper*"]),
    package_dir={"": "."},
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'modal': MODAL_REQUIRES,
    },
    python_requires='>=3.10',
)