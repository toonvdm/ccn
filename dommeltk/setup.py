from setuptools import setup, find_packages

with open("requirements.txt", "r") as req:
    requirements = req.read().splitlines()

with open("minerl_requirements.txt", "r") as minerl_req:
    minerl_requirements = minerl_req.read().splitlines()

setup(
    name="dommel",
    version="0.4.0",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={"minerl": minerl_requirements},
)
