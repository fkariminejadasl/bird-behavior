import subprocess
import sys

from pathlib import Path
from setuptools import setup, find_packages

with open("README.md", "r") as file_:
    project_description = file_.read()

with open("requirements.txt", "r") as file_:
    project_requirements = file_.read().split("\n")

setup(
    name="bird_behavior",
    version="0.1",
    description="Library for deep-learning based bird behavior",
    license="MIT",
    long_description=project_description,
    author="Fatemeh Karimi Nejadasl",
    author_email="fkariminejadasl@gmail.com",
    url="https://github.com/fkariminejadasl/bird-behavior.git",
    packages=["behavior"],
    # packages=find_packages(exclude=['docs', 'tests', "__pycache__"]),
    install_requires=project_requirements,
)
