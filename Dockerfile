# Use the NVIDIA Blackwell-optimized base
FROM nvcr.io/nvidia/pytorch:25.12-py3

# Install additional data science tools
RUN pip install --no-cache-dir \
    joblib \
    scikit-learn \
    pandas \
    matplotlib

# Set your working directory
WORKDIR /workspace
