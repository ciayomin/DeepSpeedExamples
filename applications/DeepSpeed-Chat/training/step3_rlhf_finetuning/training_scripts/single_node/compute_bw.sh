#!/bin/bash

sed -n "387362,440183p" training_log_output/vicuna_training.log | awk 'BEGIN {sum=0}{if ($25 == "MB") sum+=$24; else if ($25 =="KB") sum+=$24/1000; else sum+=$24/1000000;}END{print sum}'


sed -n "387362,440183p" training_log_output/vicuna_training.log | awk 'BEGIN {sum=0}{sum+=$33}END{print sum/(440183-387362+1)}'