from setuptools import setup, find_packages

PACKAGE_NAME = 'retrieval'
VERSION = "0.1.0"
DESCRIPTION = "Crawling package for S team."
URL = 'https://la-famille.fr'
REQUIRES_PYTHON = '>=3.6.0'

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f.readlines()]


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(include=f"{PACKAGE_NAME}.*"),
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    author="S team",
)
