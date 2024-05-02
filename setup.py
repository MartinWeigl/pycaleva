from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="pycaleva",
    version="0.8.2",
    author="Martin Weigl",
    author_email="martinweigl48@gmail.com",
    description="A framework for calibration evaluation of binary classification models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MartinWeigl/pycaleva",
    
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    keywords='calibration, classification, model, machine_learning, statistics',

    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    install_requires = ['numpy>=1.26', 'scipy>=1.13', 'scikit-learn>=1.4', 'matplotlib>=3.8', 'tqdm>=4.66', 'pandas>=2.2', 'statsmodels>=0.14', 'fpdf2>=2.7', 'ipython>=8.24'],

    project_urls={
        'Source': 'https://github.com/MartinWeigl/pycaleva',
        'Documentation': 'https://martinweigl.github.io/pycaleva/',
    },
)