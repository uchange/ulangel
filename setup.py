# coding: utf-8
import os
import re
from setuptools import setup


def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


def get_packages(package):
    """
    Return root package and all sub-packages.
    """
    return [dirpath for dirpath, dirnames, filenames in os.walk(package)
            if os.path.exists(os.path.join(dirpath, '__init__.py'))]


def get_package_data(package):
    """
    Return all files under the root package, that are not in a package themselves.
    """
    walk = [(dirpath.replace(package + os.sep, '', 1), filenames)
            for dirpath, dirnames, filenames in os.walk(package)
            if not os.path.exists(os.path.join(dirpath, '__init__.py'))]
    filepaths = []
    for base, filenames in walk:
        filepaths.extend([os.path.join(base, filename) for filename in filenames])
    return {package: filepaths}


setup(
    name='ulangel',
    version=get_version('ulangel'),
    description='All-in-one framework for NLP Transfer Learning Classification',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Célia Guanguan ZHANG',
    author_email='guanguan@uchange.co',
    url='https://github.com/uchange/ulangel',
    package_dir={'ulangel': 'ulangel'},
    packages=get_packages('ulangel'),
    package_data=get_package_data('ulangel'),
    py_modules=['ulangel'],
    zip_safe=False,
    install_requires=open('requirements.txt').read().split(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Utilities'
    ],
)
