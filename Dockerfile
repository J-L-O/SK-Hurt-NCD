FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN pip install \
    pytorch-lightning==1.1.3 \
    scikit-learn==1.2.2 \
    scipy==1.10.1 \
    pydevd-pycharm~=213.6777.50
RUN pip install lightning-bolts==0.3.0
