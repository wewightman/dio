from setuptools import setup

with open("README.md", 'r') as f:
    readme=f.read()

setup(
    name = 'dio',
    description = "Data Input and Output functions for binary datafiles and parameter files",
    long_description = readme,
    author_name = "Wren Wightman",
    author_email = "wew12@duke.edu",
    license = 'MIT',
    packages = ['dio'],
    requires=['mat73']
)