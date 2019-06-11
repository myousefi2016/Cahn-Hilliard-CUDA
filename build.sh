nvcc -gencode arch=compute_70,code=sm_70 -O3 -o main main.cu
nvcc -gencode arch=compute_70,code=sm_70 -O3 -o sweep1 sweep_1.cu
nvcc -gencode arch=compute_70,code=sm_70 -O3 -o sweep2 sweep_2.cu
