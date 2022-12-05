# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from dataclasses import dataclass

@dataclass
class Device:
    name: str = 'None'
    npus: int = sys.maxsize
    nchannels: int = sys.maxsize

Generic = Device()
A100 = Device('A100', 108, 32)
V100 = Device('V100', 80, 32)

def get_device(name):
    if name == 'A100':
        return A100
    elif name == 'V100':
        return V100
    else:
        return Generic