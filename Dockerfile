ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.11-tf2-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install nvidia-pyindex
RUN pip install --upgrade pip
RUN pip install tensorflow-addons --upgrade
RUN pip install antspyx --upgrade
RUN pip install SimpleITK --upgrade

#ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4.5.3
#ENV TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD 10000000000
ENV OMP_NUM_THREADS=2
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV OMPI_MCA_coll_hcoll_enable 0
ENV HCOLL_ENABLE_MCAST 0

# setup dependencies 
RUN apt-get update

RUN apt-get install -y cmake git
# install ants
RUN mkdir /opt/ants
WORKDIR /opt/ants

RUN git clone https://github.com/ANTsX/ANTs.git
WORKDIR /opt/ants/ANTs
RUN git checkout v2.5.0
WORKDIR /opt/ants
RUN mkdir build install
WORKDIR /opt/ants/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/opt/ants/install ../ANTs 2>&1 | tee cmake.log
RUN make -j 4 2>&1 | tee build.log
WORKDIR /opt/ants/build/ANTS-build
RUN make install 2>&1 | tee install.log
# install c3d
RUN mkdir /opt/c3d
WORKDIR /opt/c3d/
RUN wget https://downloads.sourceforge.net/project/c3d/c3d/Nightly/c3d-nightly-Linux-x86_64.tar.gz
RUN tar -xvf c3d-nightly-Linux-x86_64.tar.gz
RUN cp c3d-1.1.0-Linux-x86_64/bin/c?d /usr/local/bin/

# env
ENV PATH /opt/ants/install/bin:$PATH
RUN apt-get install -y python2.7

RUN mkdir /mist

WORKDIR /mist
ADD . /mist
