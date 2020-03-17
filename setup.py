import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wdplot",
    version="0.0.1",
    author="Wouter Dobbels",
    author_email="dobbelswouter@gmail.com",
    description="Various plotting tools on top of matplotlib.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wdobbels/wdplot",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)