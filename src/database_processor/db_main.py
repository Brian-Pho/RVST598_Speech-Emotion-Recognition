"""
This file deals with the processing of all databases into an HDF5 file. This
is the main file to run if you want to turn all of the raw databases into a
usable format for machine learning.
"""

import os

import h5py

import au_constants as auc
import db_constants as dbc
import nn_constants as nnc
import src.em_constants as emc
from cremad import read_to_melspecgram as crm
from iemocap import read_to_melspecgram as irm
from ravdess import read_to_hdf5 as rav_to_hdf5
from tess import read_to_melspecgram as trm

DB_READ_FUNCS = {
    # "CREMA-D": crm,
    # "IEMOCAP": irm,
    "RAVDESS": rav_to_hdf5,
    # "TESS": trm,
}


def main():
    # # Check if the HDF5 file already exists
    # if os.path.isfile(dbc.HDF5_DB_PATH):
    #     print("The HDF5 file already exists. Exiting.")
    #     return

    with h5py.File(dbc.HDF5_DB_PATH, "w") as hf:
        # Create the samples dataset
        samples_dset = hf.create_dataset(
            name=dbc.HDF5_SAMPLES_DSET,
            dtype=auc.WAV_DATA_TYPE,
            shape=(dbc.HDF5_NUM_SAMPLES, auc.MEL_SPECGRAM_SHAPE[0],
                   auc.MEL_SPECGRAM_SHAPE[1]),
            maxshape=(None, auc.MEL_SPECGRAM_SHAPE[0],
                      auc.MEL_SPECGRAM_SHAPE[1])
        )

        # Create the labels dataset
        labels_dset = hf.create_dataset(
            name=dbc.HDF5_LABELS_DSET,
            dtype=int,
            shape=(dbc.HDF5_NUM_SAMPLES, emc.NUM_EMOTIONS),
            maxshape=(None, emc.NUM_EMOTIONS)
        )

        # Read each database and store it in the samples and labels datasets
        master_index = 0
        for db_name, db_read_func in DB_READ_FUNCS.items():
            print("Start processing the {database} database.".format(
                database=db_name))
            master_index = db_read_func(samples_dset, labels_dset, master_index)
            print("Master index:", master_index)
            print("Finish processing the {database} database.".format(
                database=db_name))


if __name__ == "__main__":
    main()
