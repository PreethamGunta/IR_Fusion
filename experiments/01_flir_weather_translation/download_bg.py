#!/usr/bin/env python3
import os
import sys
import time

os.environ['KAGGLE_API_TOKEN'] = 'KGAT_d1d668fa7588f21aa7ca27acd0a02ae4'

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
print('Authenticated!')

print('Starting download... This may take a while for ~15GB.')
start = time.time()
api.dataset_download_files('samdazel/teledyne-flir-adas-thermal-dataset-v2', 
                           path='data/FLIR_ADAS_v2', 
                           unzip=True)
elapsed = time.time() - start
print(f'Download complete! Elapsed: {elapsed/60:.1f} minutes')
