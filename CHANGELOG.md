0.0.1
* Train and test methods of pipeline returns data
* Add PipelineStep for better management of Subpipelining
* Replace pipeline.run with pipeline.train and pipeline.test
* Remove CSVReader and add a parameter data to pipeline.run() method. 
  Data can either be a pandas dataframe or an xarray dataset.