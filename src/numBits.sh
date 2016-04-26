#!/bin/bash

python genetic.py 1000 4 -2 2 2 rose 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 8 -2 2 2 rose 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 16 -2 2 2 rose 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 32 -2 2 2 rose 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 64 -2 2 2 rose 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 128 -2 2 2 rose 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 4 -2 2 2 gold 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 8 -2 2 2 gold 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 16 -2 2 2 gold 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 32 -2 2 2 gold 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 64 -2 2 2 gold 50 0.1 >> ../output/genetic_numBits.txt
python genetic.py 1000 128 -2 2 2 gold 50 0.1 >> ../output/genetic_numBits.txt