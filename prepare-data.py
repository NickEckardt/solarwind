#!/usr/bin/env python3

import os
import ipdb
import datetime
import process_proton
import numpy as np


protondata = process_proton.get_densitydata()

os.chdir('data')
l = os.listdir()

preprocessed = None

print(l)
counter = 0
for filename in l:
    print(counter, len(l))
    counter += 1
    #ipdb.set_trace()
    split = filename.split('_')
    #print(split)
    thistime = datetime.datetime(int(split[0][0:4]), int(split[0][4:6]),
                      int(split[0][6:8]), int(split[1][0:2]))
    epoch = thistime.timestamp()
    #print(epoch)

    mintimediff = 999999999999
    mintimerow = 0
    #ipdb.set_trace()
    for i in range(0, len(protondata)):
        if mintimediff > abs(epoch - protondata[i][4]):
            mintimerow = i
            mintimediff = abs(epoch - protondata[i][4])
    
    print(mintimerow)
    if preprocessed is None:
        preprocessed = np.array([epoch, filename, protondata[mintimerow][5],
                                 protondata[mintimerow][6]])
    else:
        preprocessed = np.vstack([preprocessed,
                                np.array([epoch, filename,
                                protondata[mintimerow][5],
                                protondata[mintimerow][6]])])
    print(preprocessed[len(preprocessed) - 1])
    #ipdb.set_trace()

ipdb.set_trace()
print(preprocessed)
    


