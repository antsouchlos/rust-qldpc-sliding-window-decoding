import numpy as np


def _parse_alist_header(header):
    size = header.split()
    return int(size[0]), int(size[1])


def read_alist_file(filename):
    """
    This function reads in an alist file and creates the corresponding parity
    check matrix H. The format of alist files is described at:
    http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html
    """

    with open(filename, "r") as myfile:
        data = myfile.readlines()
        numCols, numRows = _parse_alist_header(data[0])

        H = np.zeros((numRows, numCols))

        # The locations of 1s starts in the 5th line of the file
        for lineNumber in np.arange(4, 4 + numCols):
            indices = data[lineNumber].split()
            for index in indices:
                H[int(index) - 1, lineNumber - 4] = 1

        return H.astype(np.int32)

