# Changelog


## Unereleased - 2021-DD-MM

### Added
* Logger for modules is defined in Base  ([#77](https://github.com/KIT-IAI/pyWATTS/issues/77))
* Add parameters to RMSECalculator so that it can caluclate a sliding rmse too. ([#23](https://github.com/KIT-IAI/pyWATTS/issues/28))

### Changed

### Deprecated

### Fixed


## 0.1.0 - 2021-25-03

### Added

  * Added integration tests to github actions by executing the examples in root directory ([#47](https://github.com/KIT-IAI/pyWATTS/issues/47))
  * Implementation of the profile neural network ([#71](https://github.com/KIT-IAI/pyWATTS/issues/47))
  * Imports from the init files for exception, modules, and wrapper. ([#27](https://github.com/KIT-IAI/pyWATTS/issues/27))
  * Add rolling_variance, rolling_kurtosis, rolling_skewness. ([#28](https://github.com/KIT-IAI/pyWATTS/issues/28))
  * Select the kind of groupby in the rolling function by an enum ([#28](https://github.com/KIT-IAI/pyWATTS/issues/28))

### Changed

  * Remove plot, to_csv, summary. Instead we add a callback functionality. Additional, we provide some basic callbacks. ([#16](https://github.com/KIT-IAI/pyWATTS/issues/16))
  * Change of the API  ([#16](https://github.com/KIT-IAI/pyWATTS/issues/16))
    * Use keyword arguments instead of list for determining the input of modules. keyword that start with target are only fed into the fit function.
    * Adapt the modules, so that they are compatible with the new API
    * Use xarray DataArray instead of xr.DataSets for exchaning the data
    * Remove collect step, since through the new API they are not necessary any more
    * Add ResultStep, for selecting the desired result if a moudule provides multiple ones.
    * Remove whitelister, instead columns can be selected via square brackets.
  * CalendarExtraction: Remove encoding, for each encoding and suitable feature a new feature is created. 
    E.g. month_sine. Additionally further calendar features are added. E.g. monday, tuesday, .. and cos encodings. For 
    the different features, a enum type is defined.


### Deprecated

### Fixed

  * Fixed pipeline crashing in RMSE module because of wrong time index ([#39](https://github.com/KIT-IAI/pyWATTS/issues/39))
  * Fixed bracket operator not working for steps within the pipeline ([#42](https://github.com/KIT-IAI/pyWATTS/issues/42))
  * Fixed dict objects could not be passed to pipeline ([#43](https://github.com/KIT-IAI/pyWATTS/issues/43))
  * Fixed old parameter in sample_module  ([#67](https://github.com/KIT-IAI/pyWATTS/issues/67))
  * Fixed array creation in trend extraciont ([#69](https://github.com/KIT-IAI/pyWATTS/issues/69))

## [0.0.1] - 2021-MM-DD

  * Train and test methods of pipeline returns data
  * Add PipelineStep for better management of Subpipelining
  * Replace pipeline.run with pipeline.train and pipeline.test
  * Remove CSVReader and add a parameter data to pipeline.run() method. 
    Data can either be a pandas dataframe or an xarray dataset.
