from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ac-susceptibility",
    version="0.1.0",
    description="Organize, fit and plot ac-susceptibility measurement data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Léo De Souza",
    author_email="43355143+de-souza@users.noreply.github.com",
    url="https://github.com/de-souza/ac-susceptibility",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
