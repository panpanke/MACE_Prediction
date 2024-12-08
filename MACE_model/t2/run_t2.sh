#!/bin/bash

cd /home/projects/python/MACE_model/t2

 for i in {1..428}; do
     python train_lge.py -b 16 -e 100 -p none --i $i
 done
for i in {1..61}; do
    python train_lge.py -b 16 -e 100 -p has --i $i
done

read -p "Press any key to continue..."
