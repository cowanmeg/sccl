from xml.etree import ElementTree as ET
import os

def check_threadblocks(attributes, sms):
    nthreadblocks = int(attributes['nthreadblocks'])
    valid = nthreadblocks <= sms
    if not valid:
        print(f"WARNING: Using {nthreadblocks} while GPU can support up to {sms} thread blocks")
    # print(f"Using {nthreadblocks} thread blocks")
    return valid

def check_channels(attributes, channels):
    nchannels = int(attributes['nchannels'])
    valid = nchannels <= channels
    if not valid:
        print(f"WARNING: Using {nchannels} while up to {channels} channels are allowed")
    # print(f"Using {nchannels} channels")
    return valid

def valid_resources(xml, sms, channels):
    header = open(xml, 'r').readline().strip('\n\r') + "</algo>"
    txml = ET.fromstring(header)
    attributes = txml.attrib
    return check_threadblocks(attributes, sms) and check_channels(attributes, channels)
    

def check_create(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)