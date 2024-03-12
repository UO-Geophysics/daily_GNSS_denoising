import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from numpy import genfromtxt
from tqdm import tqdm
import logging


def get_url_paths(url, ext='', params=None):
    """
    Retrieves URL paths with a specified extension from an HTML page.

    Parameters:
    - url (str): The URL of the HTML page to parse.
    - ext (str): The file extension to filter paths by.
    - params (dict): Optional parameters to include in the HTTP request.

    Returns:
    - list: List of URL paths with the specified extension.
    """
    if params is None:
        params = {}
    response = requests.get(url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
    return parent


def download_data(search_url, station, path_out):
    """
    Downloads data for a specific GPS station from a given search URL and saves it to the specified path.

    Parameters:
    - search_url (str): The base URL used for searching GPS station data.
    - station (str): The name of the GPS station.
    - path_out (str): The path where the downloaded data should be saved.
    """
    station_url = str(search_url + station + '.tenv3')

    # create station folder (if it doesn't already exist)
    station_path = path_out
    Path(station_path).mkdir(exist_ok=True)
    file_name = station_url.split('/')[-1]

    r = requests.get(station_url)
    # write it to the path and specified filename
    with open(station_path + file_name, 'wb') as f:
        f.write(r.content)


def remove_downloaded_stations(stations_all, mseed_path):
    """
    Removes stations that have already been downloaded from the list of all stations.

    Parameters:
    - stations_all (numpy.ndarray): Array containing the names of all stations.
    - mseed_path (str): The path where the downloaded station data is stored.

    Returns:
    - numpy.ndarray: Updated array with stations that have not been downloaded.
    """
    mask = np.isin(stations_all, np.unique([w[:4] for w in os.listdir(mseed_path)]))
    return stations_all[~mask]


def main():
    """
    Main function to control the download and processing of GPS data for multiple stations.
    """
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("./logs/get_pnw_data_daily.log"),
            logging.StreamHandler()
        ]
    )

    #########        Things you need to know     #############
    station_file = './ressources/pnw_stations.txt'
    search_url = 'http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/'
    path_out = '../data/raw_all_years_daily_UNR/'


    ##########################################################

    ##########       What do you want to do?     #############
    run_on_sample = False
    ##########################################################

    # read the station_file
    stations_all = genfromtxt(station_file, dtype='U')
    logging.info(f"{len(stations_all)} stations in the file")

    if run_on_sample:
        # Take only the 25 first for sample
        stations_all = stations_all[:25]

    logging.info(f"{len(stations_all)} stations selected for download")

    # Start processing
    with tqdm(total=len(stations_all)) as pbar:
        for i, station in enumerate(stations_all):
            download_data(search_url, station, path_out=path_out)
            pbar.update()
            logging.info(f"Station {station} processed, {(i/len(stations_all))*100}% done")
    logging.info("processing finished")
    return 0


if __name__ == '__main__':
    main()
