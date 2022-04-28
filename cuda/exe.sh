rm *.out
nvcc -o $1.out $1.cu
./$1.out
rm *.out
