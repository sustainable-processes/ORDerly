from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
    name="orderly",
    version="0.0.4",
    description="A wrapper for downloading ORDschema data, extracting and cleaning the data",
    license="MIT",
    long_description=long_description,
    author=["Daniel S. Wigh", "Joe Arrowsmith", "Alexander Pomberger", "Alexei A. Lapkin"],
    packages=["orderly"],
)