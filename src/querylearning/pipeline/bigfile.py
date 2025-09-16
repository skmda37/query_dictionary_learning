import argparse, os, sys, pickle
import time
import gzip, struct
from collections import namedtuple
import numpy as np
#import numba
from io import BytesIO
from collections import Counter
from typing import BinaryIO, List, Union, Callable

import torch

from querylearning.utils.timing import time_spent


def time_spent(tic1, tag='', count=1):
    toc1 = time.process_time()
    print(f"time spend on {tag} method = {(toc1 - tic1)*100./count:.2f}ms")
    return


def noop_threshold(coeff, lower=None, upper=None): 
    return coeff


HeaderDesc = namedtuple("HeaderDesc", "num_files offsets labels zipped")
ImageDesc = namedtuple("ImageDesc", "coeffs label")


class BigChunk(torch.utils.data.Dataset):

    kHeaderFmt = '>IIII'
    kSizeofHeader = struct.calcsize(kHeaderFmt)
    kOffsetFmt = '>%sq'
    kLabelFmt = '>%si'

    def __init__(
        self,
        filename: str,
        mode='rb',
        kOpen=True,
        kPickle=False
    ):
        super().__init__()
        self.numentries = 0
        self.filename = filename
        self.file = None
        self.pickle = kPickle

        if kOpen:
            # self.openfile(filename, mode)
            self.openfile(mode)

    def openfile(self, mode='rb') -> bool:
        # self.filename = filename if filename is None else filename

        if (not self.isopen) and (mode is not None):
            self.file = open(self.filename, mode)
            fin = self.file

            if (mode == 'rb'):
                self._header = self.read_header(fin)
                a, b, c, d = self._header
                self.num_files = a
                self.offset_add = b
                self.labels_add = c
                self.zipped = d

                self.offsets = self.read_offsets(
                    fin=fin,
                    offset_add=self.offset_add,
                    arr_len=self.num_files
                )
                self.labels = self.read_labels(
                    fin=fin,
                    labels_add=self.labels_add,
                    arr_len=self.num_files
                )
                self.numentries = self.num_files

        return self.isopen

    def closefile(self):
        if self.isopen:
            self.file.close()
            self.file = None
            self._header = None

    def finalize(self): self.closefile()

    @property
    def header(self) -> HeaderDesc: return self._header

    @property
    def isopen(self) -> bool: return not (self.file is None)

    def __len__(self): return self.numentries

    def __str__(self) -> str: return f"BigChunk({self.filename})"

    def __getitem__(self, index):
        offsets = self.offsets
        label = self.labels[index]

        data = self.read_file(self.file, offsets[index], offsets[index + 1])
        stream = BytesIO()  # reading to stream
        stream.write(data)
        stream.seek(0)

        # Reading coefficients
        if self.zipped:
            with gzip.GzipFile(fileobj=stream, mode='rb') as f:
                coeffs = np.load(f, allow_pickle=self.pickle)
        else:
            coeffs = np.load(stream, allow_pickle=self.pickle)
        return ImageDesc(coeffs, label)

    def read_file(self, fin: BinaryIO, offset, size):
        fin.seek(offset)
        return fin.read(size-offset)

    def read_header(self, file: BinaryIO):
        file.seek(0)
        # Byte size of header
        headersize = struct.calcsize(self.kHeaderFmt)
        # Byte content of header
        header_as_bytes = file.read(headersize)
        # Unpack byte content of header
        h = struct.unpack(self.kHeaderFmt, header_as_bytes)
        # Format header
        header = HeaderDesc(*h)
        return header

    def read_offsets(
        self,
        fin: BinaryIO,
        offset_add: int,
        arr_len: int,
        kLogging: bool = False
    ) -> tuple:
        fin.seek(offset_add)
        bytestoread = self.labels_add - self.offset_add
        if kLogging:
            print(
                f'offset add: {offset_add}, arr len: {arr_len},'
                f' bytestoread {bytestoread}'
            )
        rawoffsets = fin.read(bytestoread)
        if kLogging:
            print(f'rawoffsets: {rawoffsets}')
        offsets = struct.unpack(self.kOffsetFmt % (arr_len + 1), rawoffsets)
        return offsets

    def read_labels(
        self,
        fin: BinaryIO,
        labels_add: int,
        arr_len: int
    ) -> np.array:
        labels = None
        if labels_add != -1:
            fin.seek(labels_add)
            labels_as_bytes = fin.read()
            labels = struct.unpack(self.kLabelFmt % arr_len, labels_as_bytes)
        return np.asarray(labels)

    def write_offsets(self, fout: BinaryIO, filesizes: list) -> int:
        offsets = compute_offsets(filesizes, self.kSizeofHeader)
        self.numentries = len(filesizes)
        pos = fout.tell()
        offsets_as_bytes = struct.pack(
            self.kOffsetFmt % len(offsets),
            *offsets
        )
        fout.write(offsets_as_bytes)
        return pos

    def write_labels(self, fout: BinaryIO, labels: list) -> int:
        pos = fout.tell()
        labels_as_bytes = struct.pack(self.kLabelFmt % len(labels), *labels)
        fout.write(labels_as_bytes)
        return pos

    def write_header(
        self,
        fout: BytesIO,
        size: int,
        offset: int,
        label: int,
        zipped: int = 1
    ) -> int:
        fout.seek(0)
        header_as_bytes = struct.pack(
            self.kHeaderFmt,
            size, offset, label, zipped
        )
        fout.write(header_as_bytes)
        return fout.tell()

    def write_data(self, fout: BytesIO, data: str) -> int:
        numbytes = fout.write(data)  # # of bytes written
        return numbytes


def compute_offsets(filesizes: list, headeroffset: int) -> list:
    offsets = []
    offset = headeroffset
    for size in filesizes:
        offsets.append(offset)
        offset += size
    offsets.append(offset)
    return offsets


def reset_stream(stream: BytesIO):
    stream.truncate(0)
    stream.seek(0)


def verify1(
    i: int,
    item1,
    bigfile: BigChunk,
    xformed: List,
    threshold: Callable
) -> bool:
    item2 = bigfile[i]
    label1 = item1[1]
    label2 = item2.label
    coeff1 = threshold(xformed[i])
    coeff2 = item2.coeffs
    result = np.array_equal(coeff1, coeff2)
    result &= (label1 == label2)
    return result


def verify_bigfile(
    bigfile: BigChunk,
    dataset: Union[torch.utils.data.Dataset, List],
    xformed: List,
    threshold: noop_threshold
) -> bool:
    img1 = (xformed[1], dataset[1][1])
    img11 = bigfile[1]
    result = True
    for i, item1 in enumerate(dataset):
        result &= verify1(i, item1, bigfile, xformed, threshold)
    assert result
    return result


if __name__ == '__main__':
    pass
