
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='lpips',  
     version='0.1.4',
     author="Richard Zhang",
     author_email="rizhang@adobe.com",
     description="LPIPS Similarity metric",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/richzhang/PerceptualSimilarity",
     packages=['lpips'],
     package_data={'lpips': ['weights/v0.0/*.pth','weights/v0.1/*.pth']},
     include_package_data=True,
     install_requires=["torch>=0.4.0", "torchvision>=0.2.1", "numpy>=1.14.3", "scipy>=1.0.1", "tqdm>=4.28.1"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: BSD License",
         "Operating System :: OS Independent",
     ],
 )
