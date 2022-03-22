"""
setup.py adapted from https://github.com/kennethreitz/setup.py
"""
import io
import os
from distutils.core import setup


# Package meta-data.
name            = "ltcnn"
description     = "TensorFlow implementation of Liquid Time-Constant Neural Network (LTCNN) layers."
url             = "https://github.com/malcolmw/LTCNN"
email           = "malcolmw@mit.edu"
author          = "Malcolm C. A. White"
requires_python = ">=3"
packages        = ["ltcnn"]
required        = ["tensorflow>=2.7.0"]
license         = "MIT"

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if "README.md" is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = description


# Where the magic happens:
setup(
    name=name,
    version="0.0a0",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=author,
    author_email=email,
    python_requires=requires_python,
    url=url,
    packages=packages,
    license=license
)
