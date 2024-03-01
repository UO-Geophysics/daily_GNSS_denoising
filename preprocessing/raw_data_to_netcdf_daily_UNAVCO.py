import logging
from pathlib import Path
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xarray as xr
from numpy import genfromtxt
from tqdm import tqdm


def make_xarray_from_raw(path_out, station, station_df):
    """
    Create xarray dataset from raw GPS data for a specific station.

    Parameters:
    - path_out (str): The path where the raw GPS data is stored.
    - station (str): The name of the GPS station.
    - station_df (pd.DataFrame): DataFrame containing additional information about the stations.

    Returns:
    - xarray.Dataset: The xarray dataset created from the raw GPS data.
    """
    logging.info(f"Creating xarray for station {station}")

    station_path = path_out + station + '.csv'
    station_info = station_df[station_df['station'] == station]

    # read raw file in big dataframe
    try:
        df = dd.read_csv(station_path, comment='#', parse_dates=['Datetime']).compute()

        # drop unused columns
        df = df.drop([' Solution'], axis=1)

        # Rename site to station
        df = df.rename(
            columns={'Datetime': 'time', ' delta E': 'e', ' delta N': 'n', ' delta U': 'z', ' Std Dev E': 'std_e',
                     ' Std Dev N': 'std_n', ' Std Dev U': 'std_z'})

        # Create xarray
        ds = df.set_index(['time']).to_xarray()
        ds['time'] = pd.DatetimeIndex(ds['time'].values)

        ds = ds.expand_dims(dim={"station": [station]})
        ds = ds.assign_coords({"longitude": (('station'), station_info.longitude.values)})
        ds = ds.assign_coords({"latitude": (('station'), station_info.latitude.values)})
        ds = ds.assign_coords({"height": (('station'), station_info.elevation.values)})

        for var in ['e', 'n', 'z', 'std_e', 'std_n', 'std_z']:
            ds[var] = ds[var].assign_attrs(units='m')
            ds[var] = ds[var].astype(np.float32)

        for var in ['latitude', 'longitude']:
            ds[var] = ds[var].assign_attrs(units='deg')
        return ds.transpose()
    except:
        logging.error(f'station {station} not processed')
        return -1


def process_one_station(station, path_out, xr_path, station_df):
    """
    Process one station, create xarray dataset, and save it to a netCDF file.

    Parameters:
    - station (str): The name of the GPS station.
    - path_out (str): The path where the raw GPS data is stored.
    - xr_path (str): The path where the netCDF files will be saved.
    - station_df (pd.DataFrame): DataFrame containing additional information about the stations.
    """
    ds = make_xarray_from_raw(path_out=path_out, station=station, station_df=station_df)
    if ds != -1:
        time = pd.date_range(start="2000-01-01", end="2024-01-01", freq='1D')
        ds = ds.reindex(time=time)
        ds.to_netcdf(xr_path + station + ".nc")


def main():
    """
    Main function to control the processing of GPS data for multiple stations and saving as netCDF.

    """
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("./logs/get_pnw_data.log"),
            logging.StreamHandler()
        ]
    )

    #########        Things you need to know     #############
    station_file = './ressources/pnw_stations.txt'
    path_out = '../data/raw_CWU_daily/'
    xr_path = '../data/daily_netcdf_CWU/'
    station_df = pd.read_csv("./ressources/cascadia_station_info.csv")

    ##########################################################

    # read the station_file
    stations_all = genfromtxt(station_file, dtype='U')
    logging.info(f"{len(stations_all)} stations in the file")

    Path(xr_path).mkdir(exist_ok=True)

    # Start processing
    with tqdm(total=len(stations_all)) as pbar:
        for i, station in enumerate(stations_all):
            process_one_station(station, path_out, xr_path, station_df)
            pbar.update()
            logging.info(f"Station {station} processed, {(i / len(stations_all)) * 100}% done")
    logging.info("processing of all stations finished")
    logging.info("Open all netcdf")
    ds = xr.open_mfdataset(xr_path + '*.nc')

    # fix time index
    time = pd.date_range(start=ds.time[0].item(), end=ds.time[-1].item(), freq='1D')
    ds = ds.reindex(time=time)

    logging.info("save ds in one netcdf")
    ds.to_netcdf(f"../data/raw_daily_CWU.nc")
    logging.info("Everything is saved in netcdf")


if __name__ == '__main__':
    main()
