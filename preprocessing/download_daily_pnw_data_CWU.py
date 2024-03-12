import os
import numpy as np
import requests
from pathlib import Path
from numpy import genfromtxt
from tqdm import tqdm
import logging


def download_data(station, path_out):
    """
    Downloads GNSS data for a given station and saves it to the specified path.

    Parameters:
    - station (str): The name of the GNSS station.
    - path_out (str): The path where the downloaded data should be saved.
    """
    # get the URL listing of available YEARLY files
    station_url = f"https://web-services.unavco.org/gps/data/position/{station}/v3?analysisCenter=cwu&referenceFrame=igs14&starttime=2000-01-01&endtime=2024-01-01&report=short&dataPostProcessing=Cleaned&refCoordOption=from_analysis_center"
    # create station folder (if it doesn't already exist)
    station_path = path_out
    Path(station_path).mkdir(exist_ok=True)
    file_name = f"{station}.csv"

    # request data from url

    r = requests.get(station_url)
    if r.status_code == 200:
        # write it to the path and specified filename
        with open(station_path + file_name, 'wb') as f:
            f.write(r.content)
    else:
        logging.info(f"error {r.status_code} with station {station}")


def remove_downloaded_stations(stations_all, station_path):
    """
    Removes stations that have already been downloaded from the list of all stations.

    Parameters:
    - stations_all (numpy.ndarray): Array containing the names of all stations.
    - station_path (str): The path where the downloaded station data is stored.

    Returns:
    - numpy.ndarray: Updated array with stations that have not been downloaded.
    """
    mask = np.isin(stations_all, np.unique([w[:4] for w in os.listdir(station_path)]))
    return stations_all[~mask]


def main():
    """
    Main function to control the download and processing of GNSS data for multiple stations.
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
    path_out = '../data/raw_CWU_daily/'


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
            download_data(station, path_out=path_out)
            pbar.update()
            logging.info(f"Station {station} processed, {(i/len(stations_all))*100}% done")
    logging.info("processing finished")
    return 0


if __name__ == '__main__':
    main()
