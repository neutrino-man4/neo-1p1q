FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV TZ=Europe/Zurich
# Notice the addition of "curl"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    git vim \
    build-essential \
    cmake \
    wget openssh-client \
    curl libpython3.10-dev python3.10 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CUDA compatibility drivers

#ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/:$LD_LIBRARY_PATH
ENV BELLE2_EXEC=/cvmfs/belle.kek.jp/grid/diracos2/2.42/Linux-x86_64/diracos/bin
ENV CEPH_URL=root://ceph-node-j.etp.kit.edu
ENV EOS_MGM_URL=root://eosuser.cern.ch
ENV DEBIAN_FRONTEND=noninteractive
ENV X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates
ENV X509_VOMS_DIR=/cvmfs/grid.cern.ch/etc/grid-security/vomsdir
ENV X509_VOMSES=/cvmfs/grid.cern.ch/etc/grid-security/vomses

RUN apt-get update && apt-get install -y --no-install-recommends \
    krb5-user krb5-config \
    xrootd-client xrootd-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pandas numpy packaging typing pillow tqdm pyparsing mplhep matplotlib scipy scikit-learn pathlib h5py
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade "jax[cuda12]" jaxopt optax
RUN pip install custatevec-cu12 pennylane pennylane-lightning-gpu
RUN pip install datetime loguru tqdm awkward gpustat
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# RUN dpkg -i cuda-keyring_1.1-1_all.deb
# RUN apt-get update && apt-get install -y --no-install-recommends cuda-compat-12-6
ENV CUPY_CACHE_DIR=/tmp/cupy_cache