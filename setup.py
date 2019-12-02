import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="perceptual-similarity-richzhang", # Replace with your own username
    version="0.0.1",
    author="Richard Zhang",
    author_email="author@example.com",
    description="Learned Perceptual Image Patch Similarity (LPIPS) metric",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richzhang/PerceptualSimilarity",
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)