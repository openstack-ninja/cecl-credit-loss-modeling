#
# container support
# # As of early 2026, use the 25.12 or newer tag
# docker pull nvcr.io/nvidia/pytorch:25.12-py3
#
# run container
#docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:25.12-py3
#
## Use the NVIDIA Blackwell-optimized base
#FROM nvcr.io/nvidia/pytorch:25.12-py3
#
## Install additional data science tools
#RUN pip install --no-cache-dir \
#    joblib \
#    scikit-learn \
#    pandas \
#    matplotlib
#
#docker pull nvcr.io/nvidia/rapidsai/rapidsai:25.12-cuda13.0-py3.12
## Set your working directory
#WORKDIR /workspace
##
#
##
#docker build -t spark-ml-env .
#docker run --gpus all -it --rm -v $(pwd):/workspace spark-ml-env
#
#
#
# virtual environments to avoid conflicts
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install pandas joblib pyarrow scikit-learn matplotlib numpy

# If you want to use GPU-accelerated dataframes (RAPIDS)
#pip install cudf-cu12 --extra-index-url=https://pypi.nvidia.com
pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com

## Email support
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


# valdiate
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# disable swap unified memory
sudo swapoff -a


pip install --upgrade playwright 
playwright install



##
export FANNIE_MAE_EMAIL=mimic-hushes-03@icloud.com
export FANNIE_MAE_PASSWORD=diqji3-ziwsis-hivzYq

python3 download_data.py 

#run
python3 src/run_monte_carlo_custom_backend.py --backend cuda --n-simulations 100
