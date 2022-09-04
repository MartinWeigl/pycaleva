from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="pycaleva",
    version="0.6.1",
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
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],

    keywords='calibration, classification, model, machine_learning, statistics',

    package_dir={'': 'src'},
    packages=find_packages(where='src'),

    install_requires = ['numpy>=1.17', 'scipy>=1.3', 'matplotlib>=3.1', 'tqdm>=4.40', 'pandas>=1.3.0', 'statsmodels>=0.13.1', 'fpdf2>=2.5.0', 'ipython>=7.30.1'],

    project_urls={  # Optional
        'Source': 'https://github.com/MartinWeigl/pycaleva',
        'Documentation': 'https://martinweigl.github.io/pycaleva/',
    },
)