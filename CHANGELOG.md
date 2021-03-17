# Changelog


## [Unreleased] - 2021-MM-DD

### Added

  * Added integration tests to github actions by executing the examples in root directory ([#47](https://github.com/KIT-IAI/pyWATTS/issues/47))

### Changed

  * Remove plot, to_csv, summary. Instead we add a callback functionality. Additional, we provide some basic callbacks. ([#16](https://github.com/KIT-IAI/pyWATTS/issues/16))
  * Use keyword arguments instead of list for determining the input of modules. keyword that start with target are only fed into the fit function.  ([#16](https://github.com/KIT-IAI/pyWATTS/issues/16))

### Deprecated

### Fixed

  * Fixed pipeline crashing in RMSE module because of wrong time index ([#39](https://github.com/KIT-IAI/pyWATTS/issues/39))
  * Fixed bracket operator not working for steps within the pipeline ([#42](https://github.com/KIT-IAI/pyWATTS/issues/42))
  * Fixed dict objects could not be passed to pipeline ([#43](https://github.com/KIT-IAI/pyWATTS/issues/43))


## [0.0.1] - 2021-MM-DD

  * Train and test methods of pipeline returns data
  * Add PipelineStep for better management of Subpipelining
  * Replace pipeline.run with pipeline.train and pipeline.test
  * Remove CSVReader and add a parameter data to pipeline.run() method. 
    Data can either be a pandas dataframe or an xarray dataset.