# Piper voice training image — for Lonely's SoVITS-bootstrapped voice.
#
# Built on the NVIDIA PyTorch base which has CUDA 12.4 + torch + cuDNN
# preinstalled and matches the runtime expected by current piper1-gpl.
# Uses Python 3.10 (NVIDIA image default) which is well within Piper's
# supported 3.9–3.13 range and predates the pathlib._local reorg, so the
# old libritts_r checkpoint's pickled PosixPath objects load cleanly.
#
# Build:  docker build -f training/Dockerfile -t piper-train .
# Run:    see training/run_train_docker.ps1

FROM nvcr.io/nvidia/pytorch:24.03-py3

# Avoid interactive prompts during apt-get; install only what we need to
# build the monotonic_align cython ext (build-essential covers gcc, make).
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git \
    && rm -rf /var/lib/apt/lists/*

# Clone piper1-gpl pinned to the same commit we're using natively, so
# future image rebuilds are reproducible.
WORKDIR /opt
RUN git clone https://github.com/OHF-Voice/piper1-gpl.git
WORKDIR /opt/piper1-gpl

# Install training extras. The base image already has a compatible
# torch+CUDA — don't let the install pull a different one.
RUN pip install --no-cache-dir -e ".[train]"

# scikit-build is required by piper1-gpl's setup.py but NOT in .[train]
# extras — install it explicitly so the next step's build_ext works.
RUN pip install --no-cache-dir scikit-build cmake

# Build the cython monotonic_align extension into the source tree.
# This step bundles two builds:
#   1) the cmake-driven setup.py build_ext (builds espeakbridge), and
#   2) the smaller `monotonic_align/core.pyx` cython compile which lives
#      in its own subdirectory and has its own build script.
# The repo ships `build_monotonic_align.sh` for step 2 — run it explicitly
# because piper.train imports
# `piper.train.vits.monotonic_align.monotonic_align.core` (note the
# repeated dir), and that nested module only exists after this build.
RUN python setup.py build_ext --inplace
RUN bash build_monotonic_align.sh

# Bake in a small entrypoint wrapper that monkey-patches torch.load to
# weights_only=False BEFORE Lightning's CLI tries to load the
# pretrained libritts_r checkpoint. We need this because torch 2.6+
# defaults weights_only=True and the older Piper checkpoints contain
# pickled pathlib.PosixPath objects in their hyperparam dict, which
# aren't on torch's safe-globals allowlist.
#
# The libritts_r checkpoint is from rhasspy's official HuggingFace
# dataset — trusted source, so weights_only=False is safe.
RUN cat > /opt/piper_train_wrapper.py <<'PYEOF'
import torch
_real_load = torch.load
def _load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _real_load(*args, **kwargs)
torch.load = _load
from piper.train.__main__ import main
main()
PYEOF

# Default working dir for the user's bind-mounted dataset.
WORKDIR /workspace
ENTRYPOINT ["python", "/opt/piper_train_wrapper.py"]
