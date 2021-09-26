import setuptools

setuptools.setup(
    name="pywatts",
    version="0.1.0",
    packages=setuptools.find_packages(),

    install_requires=['scikit-learn < 1.0', 'cloudpickle', 'holidays', 'xarray>=0.19', 'numpy', 'pandas', 'matplotlib',
                      'tensorflow>=2', 'workalendar',  'statsmodels', 'tabulate'],
    extras_require={
        'dev': [
            "pytest",
            "sphinx>=4",
            "pylint",
            "pytest-cov"
        ]
    },
    author="pyWATTS-TEAM",
    author_email="pywatts-team@iai.kit.edu",
    description="A python time series pipelining project",
    keywords="preprocessing time-series machine-learning",
)
