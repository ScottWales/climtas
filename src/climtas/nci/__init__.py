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


def GadiClient(threads=1):
    """Start a Dask client on Gadi

    If run on a login node it will check the PBS resources to know how many
    CPUs and the amount of memory that is available.
    """
    global _dask_client, _tmpdir

    if _dask_client is None:
        _tmpdir = tempfile.TemporaryDirectory("dask-worker-space")

        if os.environ["HOSTNAME"].startswith("gadi-login"):
            _dask_client = dask.distributed.Client(
                n_workers=2,
                threads_per_worker=threads,
                memory_limit="1000mb",
                local_directory=_tmpdir.name,
            )
        else:
            workers = int(os.environ["PBS_NCPUS"]) // threads
            _dask_client = dask.distributed.Client(
                n_workers=workers,
                threads_per_worker=threads,
                memory_limit=int(os.environ["PBS_VMEM"]) / workers,
                local_directory=_tmpdir.name,
            )
    return _dask_client
