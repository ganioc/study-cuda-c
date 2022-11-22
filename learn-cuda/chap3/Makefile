
CUDA_PATH=/usr/local/cuda
SAMPLE_PATH=/usr/local/cuda-11.0
HOST_COMPILER ?= g++
NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${HOST_COMPILER}
NVPROF=${CUDA_PATH}/bin/nvprof

TARGET=cuda_thread_block sgemm

INCLUDES= -I${SAMPLE_PATH}/samples/common/inc 
NVCC_FLAGS=-m64 -lineinfo

IS_CUDA_11:=${expr `$(NVCC) --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode arguments
SMS = 35 37 50 52 60 61 70 75
ifeq "$(IS_CUDA_11)" "1"
SMS = 52 60 61 70 75 80
endif

$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

.PHONY: all

all: ${TARGET}

cuda_thread_block: cuda_thread_block.cu
	${NVCC} ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<

sgemm: sgemm.cu
	${NVCC} -m 64 --resource-usage  ${INCLUDES} ${ALL_CCFLAGS} ${GENCODE_FLAGS} -o $@ $<


clean:
	rm -f ${TARGET}
	rm *.nvvp
	rm *.prof


