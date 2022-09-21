from xml.etree import ElementTree as ET
import os

def check_threadblocks(attributes, sms):
    nthreadblocks = int(attributes['nthreadblocks'])
    valid = nthreadblocks <= sms
    if not valid:
        print(f"WARNING: Using {nthreadblocks} while GPU can support up to {sms} thread blocks")
    print(f"Using {nthreadblocks} thread blocks")
    return valid

def check_channels(attributes):
    nchannels = int(attributes['nchannels'])
    valid = nchannels <= CHANNELS
    if not valid:
        print(f"WARNING: Using {nchannels} while up to {CHANNELS} channels are allowed")
    print(f"Using {nchannels} channels")
    return valid

def valid_resources(xml, sms):
    header = open(xml, 'r').readline().strip('\n\r') + "</algo>"
    txml = ET.fromstring(header)
    attributes = txml.attrib
    check_threadblocks(attributes, sms)
    check_channels(attributes)

def check_create(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)