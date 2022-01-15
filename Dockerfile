FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

EXPOSE 8080

COPY ./ /workspace/temp/focus-on-you

RUN apt-get update
RUN apt-get install -y cmake build-essential wget libgl1-mesa-glx libglib2.0-0 git vim

WORKDIR /workspace/temp/focus-on-you
RUN pip install -r requirements-docker.txt

WORKDIR /workspace/temp
RUN wget http://dlib.net/files/dlib-19.9.tar.bz2
RUN tar xvf dlib-19.9.tar.bz2
WORKDIR /workspace/temp/dlib-19.9
RUN mkdir build
WORKDIR /workspace/temp/dlib-19.9/build
RUN cmake -D DLIB_USE_CUDA=1 -D USE_AVX_INSTRUCTIONS=1 -DCUDA_HOST_COMPILER=/usr/bin/gcc-7 ..
RUN cmake --build . --config Release
RUN make install
RUN ldconfig
WORKDIR /workspace/temp/dlib-19.9
RUN python setup.py install --yes USE_AVX_INSTRUCTIONS --yes DLIB_USE_CUDA 

RUN pip install face_recognition

VOLUME ["/workspace/temp/focus-on-you"]
WORKDIR /workspace/temp/focus-on-you

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8080"]