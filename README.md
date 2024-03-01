# Graph Neural Network Daily GNSS Denoising
Repository for the daily GNSS time series denoising using a graph neural network

## Preprocessing
In the preprocessing folder, you will find all the scripts and notebook to prepare the data:
* 2 download script for UNR and CWU
* 2 raw to netcdf to convert and concatenate all stations together for both processing centers
* station maintenance correction notebook to remove trend using the station maintenance logs, and remove outliers
* add tremor notebook to co-locate tremors in time and space with the stations (possibility to tune the parameters)
* resource folder with the necessary files to download and add info to the stations