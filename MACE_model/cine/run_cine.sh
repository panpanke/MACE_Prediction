#!/bin/bash

cd /home/projects/python/MACE_model/cine

 for i in {1..428}; do
     python train_cine.py -b 32 -e 30 -p none --i $i
 done
for i in {1..61}; do
    python train_cine.py -b 32 -e 50 -p has --i $i
done
read -p "Press any key to continue..."
