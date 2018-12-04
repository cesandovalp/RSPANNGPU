INC=-I/usr/local/cuda/include
NVCC=nvcc
NVCC_OPT=-std=c++11

all:
	$(NVCC) $(NVCC_OPT) hello.cu -o gpu-example

clean:
	-rm -f gpu-example