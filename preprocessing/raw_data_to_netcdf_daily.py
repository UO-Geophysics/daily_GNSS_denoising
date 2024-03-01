import logging
import dask.dataframe as dd
import pandas as pd
import xarray as xr
from numpy import genfromtxt
from tqdm import tqdm

def make_xarray_from_raw(path_out, station):
    """
    Create xarray dataset from raw GPS data for a specific station.

    Parameters:
    - path_out (str): The path where the raw GPS data is stored.
    - station (str): The name of the GPS station.
    Returns:
    - xarray.Dataset: The xarray dataset created from the raw GPS data.
    """
    logging.info(f"Creating xarray for station {station}")

    station_path = path_out + station + '.tenv3'

    # read all raw files in big dataframe
    try:
        df = dd.read_csv(station_path, sep="\s+").compute()

        # create date !!! Not handling the GPS to UTC
        df['time'] = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')

        # drop unused columns
        df = df.drop(['YYMMMDD', 'yyyy.yyyy', '__MJD', 'week', 'd', 'reflon',
                      '_e0(m)', '____n0(m)', 'u0(m)', '_ant(m)', 'sig_e(m)', 'sig_n(m)', 'sig_u(m)', '__corr_en',
                      '__corr_eu', '__corr_nu'], axis=1)

        # Rename site to station
        df = df.rename(columns={'site': 'station', '__east(m)': 'e', '_north(m)': 'n', '____up(m)': 'z',
                                '_latitude(deg)': 'latitude', '_longitude(deg)': 'longitude', '__height(m)': 'height'})

        # Create xarray
        ds = df.set_index(['time', 'station']).to_xarray()
        for var in ['e', 'n', 'z', 'height']:
            ds[var] = ds[var].assign_attrs(units='m')

        for var in ['latitude', 'longitude']:
            ds[var] = ds[var].assign_attrs(units='deg')

        ds['longitude'] = ds.longitude.mean(dim='time')
        ds['latitude'] = ds.latitude.mean(dim='time')
        ds['height'] = ds.height.mean(dim='time')
        return ds
    except:
        logging.error(f'station {station} not processed')
        return -1


def process_one_station(station, path_out, xr_path):
    """
    Process one station, create xarray dataset, and save it to a netCDF file.

    Parameters:
    - station (str): The name of the GPS station.
    - path_out (str): The path where the raw GPS data is stored.
    - xr_path (str): The path where the netCDF files will be saved.
    """
    ds = make_xarray_from_raw(path_out, station)
    if ds != -1:
        ds.to_netcdf(xr_path + station + ".nc")


def main():
    """
    Main function to process GPS data for multiple stations and save as netCDF.

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
    path_out = '../data/raw_all_years_daily_UNR/'
    xr_path = '../data/daily_netcdf_UNR/'

    ##########################################################

    # read the station_file
    stations_all = genfromtxt(station_file, dtype='U')
    logging.info(f"{len(stations_all)} stations in the file")

    # Start processing
    with tqdm(total=len(stations_all)) as pbar:
        for i, station in enumerate(stations_all):
            process_one_station(station, path_out, xr_path)
            pbar.update()
            logging.info(f"Station {station} processed, {(i / len(stations_all)) * 100}% done")
    logging.info("processing of all stations finished")
    logging.info("Open all netcdf")
    ds = xr.open_mfdataset(xr_path + '*.nc')

    # fix time index
    time = pd.date_range(start=ds.time[0].item(), end=ds.time[-1].item(), freq='1D')
    ds = ds.reindex(time=time)

    ds['longitude'] = ds.longitude.mean(dim='time')
    ds['latitude'] = ds.latitude.mean(dim='time')
    ds['height'] = ds.height.mean(dim='time')


    logging.info("save ds in one netcdf")
    ds.to_netcdf(f"../data/daily_UNR.nc")
    logging.info("Everything is saved in netcdf")


if __name__ == '__main__':
    main()
