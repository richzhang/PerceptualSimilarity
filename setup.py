import setuptools


# TODO: put in file, git, or something
VERSION = '0.1.0'


setuptools.setup(
    name='perceptualsimilarity',
    version=VERSION,
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'torch',
        'torchvision',
    ],
)
