import setuptools

dev = ["pytest",
       "sphinx>=4",
       "pylint",
       "pytest-cov"
       ]
soft_dependencies = [
    "river",
    "tensorflow>=2 ; platform_system!='Darwin'"
]

setuptools.setup(
    name="pywatts",
    version="0.3.0",
    packages=setuptools.find_packages(),

    install_requires=['scikit-learn >= 1.0', 'cloudpickle', 'holidays', 'xarray>=0.19', 'numpy', 'pandas<2.0', 'matplotlib',
                      "pywatts-pipeline@git+https://github.com/KIT-IAI/pywatts-pipeline/@main",
                      'workalendar', 'statsmodels', 'tabulate', 'tikzplotlib'],
    extras_require={
        'dev': dev,
        'all': dev + soft_dependencies
    },
    author="pyWATTS-TEAM",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Natural Language :: English",
        "Operating System :: OS Independent",

    ],
    description="A python time series pipelining project",
    keywords="preprocessing time-series machine-learning",
)
