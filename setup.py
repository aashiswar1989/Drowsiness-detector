from setuptools import setup, find_packages

PROJECT_NAME = "DrowsinessDetector"
AUTHOR = "Aashish Swarnkar"

def get_requirements(file: str) -> list:
    """
    Reads the requirements from a file and returns them as a list.
    """
    with open(file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() != "-e ."]

    return requirements


setup(
    name = PROJECT_NAME,
    version = "0.1.0",
    author = AUTHOR,
    install_requires = get_requirements('requirements_dev.txt'),
    packages = find_packages(),
    package_dir={"": "src"},
)

