import argparse
import os
import pickle
import abc
import copy
import time
import numpy as np
import gzip
import struct
from io import BytesIO
from typing import Callable, List, Union, Tuple

import torch

from querylearning.pipeline.bigfile import BigChunk, noop_threshold, reset_stream
from querylearning.pipeline.bigfile import verify_bigfile

"""
TODO:
    * remove type of fout, fin, file
"""


class BigFileBuilder:
    """Builder to apply transform (e.g. CLIP encoder) to each
        element of dataset and write to an idexable gzip file
        which is managed by BigChunk
    """

    def __init__(
        self,
        filename: str,
        xform: Callable,  # transform,
        kPickle: bool
    ):
        self._filename = filename
        self.xform = xform
        self.kPickle = kPickle

    @property
    def filename(self):
        return self._filename

    def start(self, dataset) -> Tuple[BigChunk, Callable]:
        self.num_entries = len(dataset)
        self.now = time.time()
        self.bigfile = BigChunk(
            filename=self.filename,
            mode='wb',
            kPickle=self.kPickle
        )
        assert self.bigfile.isopen
        xformed = [self.xform(x) for x, y in dataset]

        return self.bigfile, xformed

    def finalize(
        self,
        bigfile,
        offsets,
        labels,
        opt_threshold,
        zipped
    ) -> None:
        """Patch header with the offset array addresses"""
        fout = bigfile.file
        # Update header, offsets, and labels
        offset_add = bigfile.write_offsets(fout=fout, filesizes=offsets)
        labels_add = bigfile.write_labels(fout, labels)

        bigfile.write_header(
            fout,
            self.num_entries,
            offset_add,
            labels_add,
            zipped
        )
        bigfile.finalize()

    def doit(
        self,
        dataset: Union[torch.utils.data.Dataset, List],
        threshold=noop_threshold,
        verify: bool = True,
        kZip: bool = True
    ) -> None:
        bigfile, xformed = self.start(dataset)

        offsets, labels = writeBigFile(
            bigfile,
            dataset,  # Original image dataset
            xformed,  # xformed images
            threshold,
            kPickle=self.kPickle,
            kZip=kZip
        )
        self.finalize(bigfile, offsets, labels, threshold, kZip)

        if verify:
            self.bigfile.openfile(mode='rb')
            verify_bigfile(self.bigfile, dataset, xformed, threshold)


def writeBigFile(
    bigfile,
    dataset,
    xformed,
    threshold=noop_threshold,
    kPickle=False,
    kZip=True
) -> Tuple[List, List]:
    assert bigfile.isopen
    num_entries = len(dataset)
    entries = 0   # track # of entries written
    size_sum = 0  # track # of bytes written
    stream = BytesIO()
    offsets = []
    labels = []
    fout = bigfile.file

    # Writing header of file
    size_sum += bigfile.write_header(
        fout=fout,
        size=num_entries,
        offset=0,
        label=0,
        zipped=kZip
    )
    offsets, labels = writeXformed(
        bigfile=bigfile,
        size_sum=size_sum,
        dataset=dataset,
        xformed=xformed,
        threshold=threshold,
        kPickle=kPickle,
        kZip=kZip
    )
    return offsets, labels


def writeXformed(
    bigfile: BigChunk,
    size_sum: int,  # track # of bytes written
    dataset: torch.utils.data.Dataset,
    xformed: np.array,
    threshold: Callable = noop_threshold,
    kPickle: bool = False,
    kZip: bool = True
) -> Tuple[List, List]:
    num_entries = len(dataset)
    stream = BytesIO()
    offsets = []
    labels = []
    entries = 0  # track # of entries written

    for i, entry in enumerate(dataset):
        image, label = entry
        coeffs = xformed[i]
        coeffs_th = threshold(coeffs)

        # Writing data to temporary stream and getting the binary value
        reset_stream(stream)

        if kZip:
            with gzip.GzipFile(fileobj=stream, mode='wb') as f:
                np.save(f, coeffs_th, allow_pickle=kPickle)
        else:
            np.save(stream, coeffs_th, allow_pickle=kPickle)

        zipped = stream.getvalue()  # stream content in a string of bytes

        # Writing the binary data to the merged file
        sizeB = bigfile.write_data(bigfile.file, zipped)  # # bytes written
        size_sum += sizeB
        entries += 1

        offsets.append(sizeB)
        labels.append(label)

    assert entries == num_entries
    return offsets, labels


def identityXform(dataset):
    coeffs = [x[0] for x in dataset]
    return np.asarray(coeffs)


def identityXform(x): return x


if __name__ == '__main__':
    dataset = [
        (
            np.array([10.0, 10.0], dtype=np.float64),
            np.array(0, dtype=np.int32)
        ),
        (
            np.array([11.0, 11.0], dtype=np.float64),
            np.array(1, dtype=np.int32)
        ),
        (
            np.array([12.0, 12.0], dtype=np.float64),
            np.array(2, dtype=np.int32)
        ),
    ]
    print('Created dataset...')
    bigFileBuilder = BigFileBuilder(
        filename='testBigFile.dat',
        xform=identityXform,
        kPickle=True,
    )
    bigFileBuilder.doit(
        dataset=dataset
    )
    print(f'toy dataset (coeffs, label):')
    print(dataset)
    print(f'binary bigfile byte encoding:')
    bigFileBuilder.bigfile.openfile(mode='rb')
    file = bigFileBuilder.bigfile.file
    file.seek(0)
    cnt = file.read()
    print(cnt)

