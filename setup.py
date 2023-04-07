from setuptools import setup

with open("README", "r") as f:
    long_description = f.read()

setup(
    name="orderly",
    version="0.0.1",
    description="A wrapper for downloading ORD-schema data, extracting and cleaning the data",
    license="MIT",
    long_description=long_description,
    author=[
        "Daniel S. Wigh <dsw46@cam.ac.uk>",
        "Joe Arrowsmith <joearrowsmith0@gmail.com>",
        "Alexander Pomberger <ap2153@cam.ac.uk>",
        "Alexei A. Lapkin <aal35@cam.ac.uk>",
    ],
    packages=["orderly"],
)
