#!/bin/bash

./simulated_annealing CAtowns.csv 64 64 10000 4000 iter10k,sim4kdata.txt > iter10k,sim4k.out  &
./simulated_annealing CAtowns.csv 128 128 10000 16000 iter10k,sim16kdata.txt > iter10k,sim16k.out &
./simulated_annealing CAtowns.csv 64 64 50000 4000 iter50k,sim4kdata.txt > iter50k,sim4k.out &
./simulated_annealing CAtowns.csv 128 128 50000 16000 iter50k,sim16kdata.txt > iter50k,sim16k.out &
./simulated_annealing CAtowns.csv 128 128 100000 16000 iter100k,sim16kdata.txt> iter100k,sim16k.out &
./simulated_annealing CAtowns.csv 64 64 100000 4000 iter100k,sim4kdata.txt > iter100k,sim4k.out&
./simulated_annealing CAtowns.csv 64 64 500000 4000 iter500k,sim4kdata.txt > iter500k,sim4k.out&
./simulated_annealing CAtowns.csv 128 128 500000 16000 iter500k,sim16kdata.txt > iter500k,sim16k.out&
./simulated_annealing CAtowns.csv 64 64 1000000 4000 iter1mil,sim4kdata.txt > iter1mil,sim4k.out&
./simulated_annealing CAtowns.csv 128 128 1000000 16000 iter1mil,sim16kdata.txt > iter1mil,sim16k.out&
./simulated_annealing CAtowns.csv 64 64 5000000 4000 iter5mil,sim4kdata.txt > iter5mil,sim4k.out&
./simulated_annealing CAtowns.csv 128 128 5000000 16000 iter5mil,sim16kdata.txt > iter5mil,sim16k.out&

