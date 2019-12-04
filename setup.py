from setuptools import setup

def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if l]
    return requires

INSTALL_REQUIRES = parse_requirements_file('requirements.txt')

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setup(name='perceptual-similarity-pytorch',
      version='0.1',
      description='Learned Perceptual Image Patch Similarity (LPIPS) metric for Pytorch',
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      url='http://github.com/richzang/PerceptualSimilarity/',
      author='Richard Zang',
      license='BSD-2-Clause',
      packages=['perceptual_similarity'],
      include_package_data=True,
      install_requires=INSTALL_REQUIRES,
      classifiers=[
          "License :: OSI Approved :: BSD License",
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 3",
          ],
      zip_safe=False)
