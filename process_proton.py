#!/usr/bin/env python3

import numpy as np
import time
import ipdb


def get_densitydata():

    data = np.genfromtxt('protondata.txt', dtype=[int, int, int, int, float,
                                                  float, float])

    #units of last two columns: nH, km/s
    #dist earth to sun: 149,597,870

    #Add time_offset to all timestamps
    time_offset = 820453631
    #ipdb.set_trace()

    for i in range(0, len(data)):
        data[i][4] += time_offset

    #ipdb.set_trace()
    #time.gmtime(123456)

    return data




if __name__ == '__main__':
    preprocess()


