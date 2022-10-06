import datetime
import logging
import re
from typing import TextIO as File
from typing import Tuple

import numpy as np

EVENT_CHANNEL = 'EDF Annotations'
log = logging.getLogger(__name__)


class EDFEndOfData(BaseException):
    """Costumer Exception.
    """
    pass


def tal(tal_str: str) -> list:
    """Return a list with (onset, duration, annotation) tuples for an EDF+ TAL
    steam.
    """
    exp = r'(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
          r'(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
          r'(\x14(?P<annotation>[^\x00]*))?' + \
          r'(?:\x14\x00)'

    return [(
        float(dic['onset']),
        float(dic['duration']) if dic['duration'] else 0.,
        dic['annotation'].split('\x14') if dic['annotation'] else []
    )
        for dic
        in map(lambda m: m.groupdict(), re.finditer(exp, tal_str))
    ]


def edf_header(f: File) -> dict:
    h = {}
    assert f.tell() == 0  # check file position
    assert f.read(8) == '0       '

    # recording info
    h['local_subject_id'] = f.read(80).strip()
    h['local_recording_id'] = f.read(80).strip()

    # parse timestamp
    day, month, year = [int(x) for x in re.findall(r'(\d+)', f.read(8))]
    hour, minute, sec = [int(x) for x in re.findall(r'(\d+)', f.read(8))]
    h['date_time'] = str(
        datetime.datetime(year + 2000, month, day, hour, minute, sec)
    )

    # misc
    header_ntypes = int(f.read(8))
    subtype = f.read(44)[:5]
    h['EDF+'] = subtype in ['EDF+C', 'EDF+D']
    h['contiguous'] = subtype != 'EDF+D'
    h['n_records'] = int(f.read(8))
    h['record_length'] = float(f.read(8))  # in seconds
    nchannels = h['n_channels'] = int(f.read(4))

    # read channel info
    channels = list(range(h['n_channels']))
    h['label'] = [f.read(16).strip() for _ in channels]
    h['transducer_type'] = [f.read(80).strip() for _ in channels]
    h['units'] = [f.read(8).strip() for _ in channels]
    h['physical_min'] = np.asarray([float(f.read(8)) for _ in channels])
    h['physical_max'] = np.asarray([float(f.read(8)) for _ in channels])
    h['digital_min'] = np.asarray([float(f.read(8)) for _ in channels])
    h['digital_max'] = np.asarray([float(f.read(8)) for _ in channels])
    h['prefiltering'] = [f.read(80).strip() for _ in channels]
    h['n_samples_per_record'] = [int(f.read(8)) for _ in channels]
    f.read(32 * nchannels)  # reserved

    assert f.tell() == header_ntypes
    return h


class BaseEDFReader:
    def __init__(self, file: File):
        self.gain = None
        self.phys_min = None
        self.dig_min = None
        self.header = None
        self.file = file

    def read_header(self):
        self.header = h = edf_header(self.file)

        # calculate ranges for rescalling
        self.dig_min = h['digital_min']
        self.phys_min = h['physical_min']
        phys_range = h['physical_max'] - h['physical_min']
        dig_range = h['digital_max'] - h['digital_min']
        assert np.all(phys_range > 0)
        assert np.all(dig_range > 0)
        self.gain = phys_range / dig_range

    def read_raw_record(self) -> list:
        """Read a record with data and return a list containing arrays with
        raw bytes.
        """
        result = []
        for nsamp in self.header['n_samples_per_record']:
            samples = self.file.read(nsamp * 2)
            if len(samples) != nsamp * 2:
                raise EDFEndOfData
            result.append(samples)
        return result

    def convert_record(self, raw_record: list) -> Tuple[float, list, list]:
        """Convert a raw record to a (time, signal, events) tuple based on
        information in the header.
        """
        h = self.header
        dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
        time = float('nan')
        signals, events = [], []

        for i, samples in enumerate(raw_record):
            if h['label'][i] == EVENT_CHANNEL:
                ann = tal(samples)
                time = ann[0][0]
                events.extend(ann[1:])
            else:
                # 2-byte little-endian integers
                dig = np.fromstring(samples, '<i2').astype(np.float32)
                phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
                signals.append(phys)

        return time, signals, events

    def read_record(self) -> Tuple[float, list, list]:
        return self.convert_record(self.read_raw_record())

    def records(self):
        """
        Record generator.
        """
        try:
            while True:
                yield self.read_record()
        except EDFEndOfData:
            pass
