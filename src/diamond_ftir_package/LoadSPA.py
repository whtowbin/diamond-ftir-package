# %%
import pathlib
import numpy as np
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
import re
import xarray as xr

from .Spectrum_obj import Spectrum

# %%


def _read_header(fid, pos, spa_name):
    """
    read spectrum/ifg/series header

    Parameters
    ----------
    fid : BufferedReader
        The buffered binary stream.

    pos : int
        The position of the header (see Notes).

    spa_name : str
        The name of the spa file.

    Returns
    -------
        dict, int
        Dictionary and current position in file

    Notes
    -----
        So far, the header structure is as follows:

        - starts with b'\x01' , b'\x02', b'\x03' ... maybe indicating the header "type"
        - nx (UInt32): 4 bytes behind
        - xunits (UInt8): 8 bytes behind. So far, we have the following correspondence:

            * `x\01` : wavenumbers, cm-1
            * `x\02` : datapoints (interferogram)
            * `x\03` : wavelength, nm
            * `x\04' : wavelength, um
            * `x\20' : Raman shift, cm-1

        - data units (UInt8): 12 bytes behind. So far, we have the following
          correspondence:

            * `x\11` : absorbance
            * `x\10` : transmittance (%)
            * `x\0B` : reflectance (%)
            * `x\0C` : Kubelka_Munk
            * `x\16` :  Volts (interferogram)
            * `x\1A` :  photoacoustic
            * `x\1F` : Raman intensity

        - first x value (float32), 16 bytes behind
        - last x value (float32), 20 bytes behind
        - ... unknown
        - scan points (UInt32), 28 bytes behind
        - zpd (UInt32),  32 bytes behind
        - number of scans (UInt32), 36 bytes behind
        - ... unknown
        - number of background scans (UInt32), 52 bytes behind
        - ... unknown
        - collection length in 1/100th of sec (UIint32), 68 bytes behind
        - ... unknown
        - reference frequency (float32), 80 bytes behind
        - ...
        - optical velocity (float32), 188 bytes behind
        - ...
        - spectrum history (text), 208 bytes behind

        For "rapid-scan" srs files:

        - series name (text), 938 bytes behind
        - collection length (float32), 1002 bytes behind
        - last y (float 32), 1006 bytes behind
        - first y (float 32), 1010 bytes behind
        - ny (UInt32), 1026
        - ... y unit could be at pos+1030 with 01 = minutes ?
        - history (text), 1200 bytes behind (only initila hgistopry.
           When reprocessed, updated history is at the end of the file after the
           b`\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF` sequence
    """

    out = {}
    # determine the type of file
    fid.seek(0)
    bytes = fid.read(18)
    if bytes == b"Spectral Data File":
        filetype = "spa, spg"
    elif bytes == b"Spectral Exte File":
        filetype = "srs"

    # nx
    fid.seek(pos + 4)
    out["nx"] = np.fromfile(fid, "uint32", count=1)

    # xunits
    fid.seek(pos + 8)
    key = np.fromfile(fid, dtype="uint8", count=1)
    if key == 1:
        out["xunits"] = "cm^-1"
        out["xtitle"] = "wavenumbers"
    elif key == 2:
        out["xunits"] = None
        out["xtitle"] = "data points"
    elif key == 3:  # pragma: no cover
        out["xunits"] = "nm"
        out["xtitle"] = "wavelengths"
    elif key == 4:  # pragma: no cover
        out["xunits"] = "um"
        out["xtitle"] = "wavelengths"
    elif key == 32:  # pragma: no cover
        out["xunits"] = "cm^-1"
        out["xtitle"] = "raman shift"
    else:  # pragma: no cover
        out["xunits"] = None
        out["xtitle"] = "xaxis"
        print("The nature of x data is not recognized, xtitle is set to 'xaxis'")
        print(spa_name)

    # data units
    fid.seek(pos + 12)
    key = np.fromfile(fid, dtype="uint8", count=1)
    if key == 17:
        out["units"] = "absorbance"
        out["title"] = "absorbance"
    elif key == 16:  # pragma: no cover
        out["units"] = "percent"
        out["title"] = "transmittance"
    elif key == 11:  # pragma: no cover
        out["units"] = "percent"
        out["title"] = "reflectance"
    elif key == 12:  # pragma: no cover
        out["units"] = None
        out["title"] = "log(1/R)"
    elif key == 20:  # pragma: no cover
        out["units"] = "Kubelka_Munk"
        out["title"] = "Kubelka-Munk"
    elif key == 21:
        out["units"] = None
        out["title"] = "reflectance"
    elif key == 22:
        out["units"] = "V"
        out["title"] = "detector signal"
    elif key == 26:  # pragma: no cover
        out["units"] = None
        out["title"] = "photoacoustic"
    elif key == 31:  # pragma: no cover
        out["units"] = None
        out["title"] = "Raman intensity"
    else:  # pragma: no cover
        out["units"] = None
        out["title"] = "intensity"
        print("The nature of data is not recognized, title set to 'Intensity'")
        print(spa_name)

    # firstx, lastx
    fid.seek(pos + 16)
    out["firstx"] = np.fromfile(fid, "float32", 1)
    fid.seek(pos + 20)
    out["lastx"] = np.fromfile(fid, "float32", 1)
    fid.seek(pos + 28)

    out["scan_pts"] = np.fromfile(fid, "uint32", 1)
    fid.seek(pos + 32)
    out["zpd"] = np.fromfile(fid, "uint32", 1)
    fid.seek(pos + 36)
    out["nscan"] = np.fromfile(fid, "uint32", 1)
    fid.seek(pos + 52)
    out["nbkgscan"] = np.fromfile(fid, "uint32", 1)
    fid.seek(pos + 68)
    out["collection_length"] = np.fromfile(fid, "uint32", 1)
    fid.seek(pos + 80)
    out["reference_frequency"] = np.fromfile(fid, "float32", 1)
    fid.seek(pos + 188)
    out["optical_velocity"] = np.fromfile(fid, "float32", 1)

    if filetype == "spa, spg":
        out["history"] = _readbtext(fid, pos + 208, None)

    if filetype == "srs":
        if out["nbkgscan"] == 0:
            # an interferogram in rapid scan mode
            if out["firstx"] > out["lastx"]:
                out["firstx"], out["lastx"] = out["lastx"], out["firstx"]

        out["name"] = _readbtext(fid, pos + 938, 256)
        fid.seek(pos + 1002)
        out["collection_length"] = np.fromfile(fid, "float32", 1) * 60
        fid.seek(pos + 1006)
        out["lasty"] = np.fromfile(fid, "float32", 1)
        fid.seek(pos + 1010)
        out["firsty"] = np.fromfile(fid, "float32", 1)
        fid.seek(pos + 1026)
        out["ny"] = np.fromfile(fid, "uint32", 1)
        #  y unit could be at pos+1030 with 01 = minutes ?
        out["history"] = _readbtext(fid, pos + 1200, None)

        if _readbtext(fid, pos + 208, 256)[:10] == "Background":
            # it is the header of a background
            out["background_name"] = _readbtext(fid, pos + 208, 256)[10:]
    return out


def _getintensities(fid, pos) -> np.ndarray:
    # get intensities from the 03 (spectrum)
    # or 66 (sample ifg) or 67 (bg ifg) key,
    # returns a ndarray

    fid.seek(pos + 2)  # skip 2 bytes
    intensity_pos = np.fromfile(fid, "uint32", 1)[0]
    fid.seek(pos + 6)
    intensity_size = np.fromfile(fid, "uint32", 1)[0]
    nintensities = int(intensity_size / 4)

    # Read and return spectral intensities
    fid.seek(intensity_pos)
    return np.fromfile(fid, "float32", int(nintensities))


def _readbtext(fid, pos, size) -> str:
    # Read some text in binary file of given size. If size is None, the etxt is read
    # until b\0\ is encountered.
    # Returns utf-8 string
    fid.seek(pos)
    if size is None:
        btext: Literal[b""] = b""
        while fid.read(1) != b"\x00":
            btext += fid.read(1)
    else:
        btext = fid.read(size)
    btext = re.sub(pattern=b"\x00+", repl=b"\n", string=btext)

    if btext[:1] == b"\n":
        btext = btext[1:]

    if btext[-1:] == b"\n":
        btext = btext[:-1]

    try:
        text = btext.decode(encoding="utf-8")  # decode btext to string
    except UnicodeDecodeError:
        try:
            text = btext.decode(encoding="latin_1")
        except UnicodeDecodeError:  # pragma: no cover
            text = btext.decode(encoding="utf-8", errors="ignore")
    return text


def _nextline(pos):
    # reset current position to the beginning of next line (16 bytes length)
    return 16 * (1 + pos // 16)


def Load_SPA(filepath: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Process a spa file and return the spectra, wavelength, and title.m
    Modified from code written for SpectoChemPy
    """

    with open(filepath, "rb") as f:
        f.seek(30)
        spa_name = f.read()
        f.seek(296)
        timestamp = np.fromfile(f, dtype="uint32", count=1)
        acqdate = datetime(1899, 12, 31, 0, 0, tzinfo=timezone.utc) + timedelta(
            seconds=int(timestamp)
        )
        acquisitiondate = acqdate

        # From hex 120 = decimal 304, the spectrum is described
        # by a block of lines starting with "key values",
        # for instance hex[02 6a 6b 69 1b 03 82] -> dec[02 106  107 105 27 03 130]
        # Each of these lines provides positions of data and metadata in the file:
        #
        #     key: hex 02, dec  02: position of spectral header (=> nx,
        #                                 firstx, lastx, nscans, nbkgscans)
        #     key: hex 03, dec  03: intensity position
        #     #     key: hex 04, dec  04: user text position (custom info, can be present
        #                           several times. The text length is five bytes later)
        #     key: hex 1B, dec  27: position of History text, The text length
        #                           is five bytes later
        #     key: hex 53, dec  83: probably not a position, present when 'Retrieved from library'
        #     key: hex 64, dec 100: ?
        #     key: hex 66  dec 102: sample interferogram
        #     key: hex 67  dec 103: background interferogram
        #     key: hex 69, dec 105: ?
        #     key: hex 6a, dec 106: ?
        #     key: hex 80, dec 128: ?
        #     key: hex 82, dec 130: position of 'Experiment Information', The text length
        #                           is five bytes later. The block gives Experiment filename (at +10)
        #                           Experiment title (+90), custom text (+254), accessory name (+413)
        #     key: hex 92, dec 146: position of 'custom infos', The text length
        #                           is five bytes later.
        #
        # The line preceding the block start with '01' or '0A'
        # The lines after the block generally start with '00', except in few cases where
        # they start by '01'. In such cases, the '53' key is also present
        # (before the '1B').

        # scan "key values"
        pos = 304
        return_ifg = None
        spa_comments = []  # several custom comments can be present
        # info: dict[str, Any]
        # intensities: np.ndarray

        while "continue":
            f.seek(pos)
            key = np.fromfile(f, dtype="uint8", count=1)

            if key == 2:
                # read the position of the header
                f.seek(pos + 2)
                pos_header = np.fromfile(f, dtype="uint32", count=1)[0]
                info = _read_header(f, pos_header, filepath)

            elif key == 3 and return_ifg is None:
                intensities = _getintensities(f, pos)

            elif key == 00 or key == 1:
                break

            pos += 16

        Spectrum_name = spa_name.split(b"\x00")[0].strip().decode("latin-1", "replace")
        filename = pathlib.Path(filepath).name

        try:
            Metadata = {
                "Filename": filename,
                "Title": Spectrum_name,
                "Acquisition_Date": acquisitiondate,
                "spa_comments": spa_comments,
                **info,
            }
            x_data = np.linspace(info["firstx"][0], info["lastx"][0], info["nx"][0])
            # Data = {"X": x_data, "Y": intensities}
            # return Data, Metadata

            SPA_Spectrum = Spectrum(
                X=x_data,
                Y=intensities,
                X_Unit=Metadata["xunits"],
                Y_Unit=Metadata["units"],
                metadata=Metadata,
            )
            return SPA_Spectrum
        except Exception as e:
            print(f"An exception occured: {e}")


def parse_coordinates(string: str) -> tuple[float, float]:
    """Function to parse the coordinates from the title of an spa file in the format autogenerated when exporting a map from OMNIC.
      Function generated with assistance from github copilot chat
    Args:
        string (str): _description_

    Returns:
        tuple[float, float]: _description_
    """
    pattern = re.compile(
        r"Position\s*\(X,Y\):\s*(?P<x>-?\d+(\.\d+)?),\s*(?P<y>-?\d+(\.\d+)?)"
    )
    match = pattern.search(string)
    if match:
        x = float(match.group("x"))
        y = float(match.group("y"))
        return x, y
    else:
        print("No match found in:")
        print(string)
        return None


# %%
# Write Funtion to process the files in multiple ways.
# One way extracts the positition from the title of an SPA file
# Another way  explicitly constructs a map from the shape of the array and the step size in x and y


def Load_SPA_Map(dir_path: str):
    #     """
    #     Process a directory of spa files and return the an array of spectra, wavelength, position, and spetrum title.
    #     Modified from code written for SpectoChemPy
    #     Assumes all spectra are equal length.
    #     Needs a shape to be input as XY dimensions.
    #     Needs  XY increments to be given. Step Size is assumed to be the same in x and y. I should improve this to handle either.
    #     """

    # # load maps using Pathlib Glob *.spa for a given directory
    # # Map an empty numpy array that is the size of the full array.
    # # Loop through the files and load the spectra into the array.

    # Save the array as a more memory efficient structure than spa files
    # write funtions to interate through the array and appply the individual operations such as fitting and baseline correction.

    # Function to parse the coordinates from the title of the spa file Generated in github copilot chat

    # %%

    directory = pathlib.Path(dir_path)
    # get filenames in directory
    filenames = directory.glob("*.spa")

    coordinates = []  # list of tuples with x and y coordinates
    spectra_intensties = []  # list of numpy arrays with spectra intensities
    aquisition_dates = []
    intensity_units = None
    wave_units = None

    # loop through files and load spectra then add the intenties and metadata to the lists to be made into an array
    first = True
    try:
        for idx, name in enumerate(filenames):
            Spectrum = Load_SPA(name)
            Spectrum.metadata["Position"] = parse_coordinates(
                Spectrum.metadata["Title"]
            )
            coordinates.append(Spectrum.metadata["Position"])
            spectra_intensties.append(Spectrum.Y)
            aquisition_dates.append(Spectrum.metadata["Acquisition_Date"])

            if first:  # get the units and wavenumber from the first spectrum
                first = False
                spectra_wavenumber = Spectrum.X
                intensity_units = Spectrum.metadata["units"]
                wave_units = Spectrum.metadata["xtitle"]

        # Generate XArray from data
        # xr_spectra = xr.DataArray(
        #     [coordiantes[:,0], coordiantes[:,1], spectra_intensties,
        #     dims=("X", "Y", "Wavenumber", "Aquisition_Date"),
        #     coords={
        #         "X": [x[0] for x in coordinates],
        #         "Y": [x[1] for x in coordinates],
        #         "Wavenumber": spectra_wavenumber,
        #         "Aquisition_Date": aquisition_dates,
        #     },
        # )
        # xr_spectra.attrs["Intensity_Units"] = intensity_units
        # xr_spectra.attrs["Wavenumber_Units"] = wave_units
        # return xr_spectra

        # Generate XArray from data

        coordinate_array = np.array(coordinates)
        x_points = coordinate_array[:, 0]
        y_points = coordinate_array[:, 1]
        x_coordinates = np.sort(np.unique(x_points))  # Subratact Min Vale to place corner at 0.
        y_coordinates = np.sort(np.unique(y_points))

        wavenumber_array = np.array(spectra_wavenumber)
        spectra_intensties_array = np.array(spectra_intensties)

        numpy_time = np.array(aquisition_dates)

        reshaped_array = np.empty(
            (len(x_coordinates), len(y_coordinates), len(wavenumber_array))
        )
        time_array = np.empty((len(x_coordinates), len(y_coordinates), 1))
        # create a 3D array with the spectra based on their x and y coordinates
        for idx in range(spectra_intensties_array.shape[0]):
            x_coord = x_points[idx]
            y_coord = y_points[idx]
            x_idx = np.where(x_coordinates == x_coord)[0][0]
            y_idx = np.where(y_coordinates == y_coord)[0][0]
            reshaped_array[x_idx, y_idx, :] = spectra_intensties_array[idx, :]
            # time_array[x_idx, y_idx, :] = aquisition_dates[idx]

        # Define xarray for the map data

        # then define the data
        data = {"spectra": (["x", "y", "wn"], reshaped_array)}

        print(reshaped_array.shape)
        # coords = {
        #     "x": (["x"], x_coordinates),
        #     "y": (["y"], y_coordinates),
        #     "wn": (["wn"], wavenumber_array),
        # }

        coords = {
            "x": x_coordinates,
            "y": y_coordinates,
            "wn": wavenumber_array,
        }

        dataset = xr.Dataset(
            data,
            coords,
        )
        return dataset

    except Exception as e:
        print(f"An exception occured: {e}")
        print(
            "Check to make sure the directory doesn't contain any background spectra or spa files that are not from intended the map"
        )




def load_SPA_Directory(dir_path: str):

    directory = pathlib.Path(dir_path)
    # get filenames in directory
    filenames = directory.glob("*.spa")
    
    coordinates = []  # list of tuples with x and y coordinates
    spectra_intensties = []  # list of numpy arrays with spectra intensities
    aquisition_dates = []
    intensity_units = None
    wave_units = None




# %%
# List of potential functions for map processing
# crop - XY
# mask based on conditional
# interpolate
# normalize
# baseline correction

# Funtion To filter and mask based on conditional.

# Sumarize data
# - peak fitting
# - peak area integration
# - peak height

# One way to handle various processed maps is to add them to a dictionary of xarray.arrays that can be called individually

# %%


# %%
# apply baseline alogorithm to each spectrum in the xarray
def baseline_correction(xarray, algorithm):
    """Function to apply a baseline correction algorithm to each spectrum in an xarray
    Args:
        xarray (xarray): xarray of spectra
        algorithm (str): name of the algorithm to use
    Returns:
        xarray: xarray of spectra with the baseline corrected
    """
    # check to make sure the algorithm is valid
    if algorithm == "ALS":
        pass
    elif algorithm == "Modpoly":
        pass
    elif algorithm == "Polyfit":
        pass
    elif algorithm == "Rubberband":
        pass
    elif algorithm == "SNIP":
        pass
    elif algorithm == "Zhangfit":
        pass
    else:
        print("Please select a valid baseline correction algorithm")
        return None

    # apply the algorithm to each spectrum


def spa_to_xarray(filepath):

    Spectrum = Load_SPA(filepath)
    #sample_name = filepath.name.split("-")[0]
    wavenumber = Spectrum.X
    intensities = Spectrum.Y

    unit_names = {"x": "um", "y": "um", "wn": "cm^-1", "data": "absorbance"}
    unit_long_names = {"x": "microns", "y": "microns", "wn": "wavenumbers", "data": "absorbance"}
    #metadata = {"sample_name" : sample_name, "unit_names": unit_names, "unit_long_names": unit_long_names}
    metadata = {"unit_names": unit_names, "unit_long_names": unit_long_names}


    DataArray = xr.DataArray(
        [[intensities]],
        dims=("y", "x", "wn"),
        coords={"y": np.arange(1), "x": np.arange(1), "wn": wavenumber},
        attrs= metadata
    )
    return DataArray

