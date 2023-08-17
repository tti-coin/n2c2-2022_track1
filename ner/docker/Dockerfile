FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN pip3 uninstall -y confection spacy thinc && pip3 install -U pydantic==2.1.0 && \
    pip3 install sentencepiece==0.1.99 transformers==4.17.0
