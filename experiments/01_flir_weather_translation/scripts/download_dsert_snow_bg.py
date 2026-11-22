#!/usr/bin/env python3
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_dsert_snow import main
import argparse
sys.argv = ['download_dsert_snow.py', '--workers', '8']
main()
