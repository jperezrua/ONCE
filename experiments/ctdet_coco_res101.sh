python main.py ctdet --exp_id coco_res101 --arch res_101 --batch_size 128 --master_batch 5 --lr 3.75e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16
#srun --gres gpu:8 --cpus-per-task 12 --partition long --pty /bin/bash

