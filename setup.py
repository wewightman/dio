from setuptools import setup

with open("README.md", 'r') as f:
    readme = f.read()

# run setup tools
setup(
    name="pyusel-dio",
    description="data io wrappers for binary and matlab datatypes",
    author="Wren Wightman",
    author_email="wew12@duke.edu",
    long_description=readme,
    packages=["dio"],
    version="0.0.0"
)