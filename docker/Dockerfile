# Basic image
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubi8

# Install Python3 and pip3
RUN yum update -y && yum install -y python3 python3-pip && yum clean all

# Install Miniconda
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -rf /tmp/miniconda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Initialize Conda in bash mode
RUN conda init bash

# Install Mamba from Conda-Forge
RUN conda install -c conda-forge mamba

# Install PyTorch and torchvision using Mamba
RUN mamba install -y pytorch torchvision torchaudio cudatoolkit -c pytorch

# Copy all files under the current folder to the container
COPY . /workspace

# Set the working directory
WORKDIR /workspace

# Install additional dependencies
RUN /bin/bash -c "source install.sh cu113 dance-env"

# Command to activate Conda environment and start Python
CMD ["/bin/bash", "-c", "source activate dance-env && python"]
