#!/usr/bin/env python
# Copyright 2020 Scott Wales
# author: Scott Wales <scott.wales@unimelb.edu.au>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NCI Specific functions and utilities
"""

import dask.distributed
import os
import tempfile

_dask_client = None
_tmpdir = None


def Client(threads=1, malloc_trim_threshold=None):
    """Start a Dask client at NCI

    An appropriate client will be started for the current system

    Args:
        threads: Number of threads per worker process. The total number of
            workers will be ncpus/threads, so that each thread gets its own
            CPU
        malloc_trim_threshold: Threshold for automatic memory trimming. Can be
            either a string e.g. '64kib' or a number of bytes e.g. 65536.
            Smaller values may reduce out of memory errors at the cost of
            running slower

    https://distributed.dask.org/en/latest/worker.html?highlight=worker#automatically-trim-memory
    """

    if os.environ["HOSTNAME"].startswith("ood"):
        return OODClient(threads, malloc_trim_threshold)
    else:
        return GadiClient(threads, malloc_trim_threshold)


def OODClient(threads=1, malloc_trim_threshold=None):
    """Start a Dask client on OOD

    This function is mostly to be consistent with the Gadi version

    Args:
        threads: Number of threads per worker process. The total number of
            workers will be ncpus/threads, so that each thread gets its own
            CPU
        malloc_trim_threshold: Threshold for automatic memory trimming. Can be
            either a string e.g. '64kib' or a number of bytes e.g. 65536.
            Smaller values may reduce out of memory errors at the cost of
            running slower

    https://distributed.dask.org/en/latest/worker.html?highlight=worker#automatically-trim-memory
    """
    global _dask_client, _tmpdir

    env = {}

    if malloc_trim_threshold is not None:
        env["MALLOC_TRIM_THRESHOLD_"] = str(
            dask.utils.parse_bytes(malloc_trim_threshold)
        )

    if _dask_client is None:
        try:
            # Works in sidebar and can follow the link
            dask.config.set(
                {
                    "distributed.dashboard.link": f'/node/{os.environ["host"]}/{os.environ["port"]}/proxy/{{port}}/status'
                }
            )
        except KeyError:
            # Works in sidebar, but can't follow the link
            dask.config.set({"distributed.dashboard.link": "/proxy/{port}/status"})

        _dask_client = dask.distributed.Client(threads_per_worker=threads, env=env)

    return _dask_client


def GadiClient(threads=1, malloc_trim_threshold=None):
    """Start a Dask client on Gadi

    If run on a compute node it will check the PBS resources to know how many
    CPUs and the amount of memory that is available.

    If run on a login node it will ask for 2 workers each with a 1GB memory
    limit

    Args:
        threads: Number of threads per worker process. The total number of
            workers will be $PBS_NCPUS/threads, so that each thread gets its own
            CPU
        malloc_trim_threshold: Threshold for automatic memory trimming. Can be
            either a string e.g. '64kib' or a number of bytes e.g. 65536.
            Smaller values may reduce out of memory errors at the cost of
            running slower

    https://distributed.dask.org/en/latest/worker.html?highlight=worker#automatically-trim-memory
    """
    global _dask_client, _tmpdir

    env = {}

    if malloc_trim_threshold is not None:
        env["MALLOC_TRIM_THRESHOLD_"] = str(
            dask.utils.parse_bytes(malloc_trim_threshold)
        )

    if _dask_client is None:
        _tmpdir = tempfile.TemporaryDirectory("dask-worker-space")

        if os.environ["HOSTNAME"].startswith("gadi-login"):
            _dask_client = dask.distributed.Client(
                n_workers=2,
                threads_per_worker=threads,
                memory_limit="1000mb",
                local_directory=_tmpdir.name,
                env=env,
            )
        else:
            workers = int(os.environ["PBS_NCPUS"]) // threads
            _dask_client = dask.distributed.Client(
                n_workers=workers,
                threads_per_worker=threads,
                memory_limit=int(os.environ["PBS_VMEM"]) / workers,
                local_directory=_tmpdir.name,
                env=env,
            )
    return _dask_client
