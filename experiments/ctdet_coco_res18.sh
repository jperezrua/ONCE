################# COCO FULL RES-18
python main.py ctdet --exp_id coco_res18 --arch res_18 --batch_size 114 --master_batch 18 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16

# TEST
python test.py ctdet --exp_id coco_res18_nohc --arch res_18 --keep_res --resume
python test.py ctdet --exp_id coco_res18_nohc --arch res_18 --keep_res --resume --flip_test 
python test.py ctdet --exp_id coco_res18_nohc --arch res_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5

################# COCO FULL RES-101
python main.py ctdet --exp_id coco_res101_nohc --arch res_101 --batch_size 84 --master_batch 5 --lr 3.75e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --dataset coco --head_conv 0

# TEST
python test.py ctdet --exp_id coco_res101_nohc --arch res_101 --keep_res --resume
python test.py ctdet --exp_id coco_res101_nohc --arch res_101 --keep_res --resume --flip_test 
python test.py ctdet --exp_id coco_res101_nohc --arch res_101 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5

################ COCO BASE RES-18
python main.py ctdet --exp_id coco_base_res18_nohc --arch res_18 --batch_size 114 --master_batch 18 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16 --dataset coco_base

################ COCO BASE RES-101
python main.py ctdet --exp_id coco_base_res101_nohc --arch res_101 --batch_size 84 --master_batch 5 --lr 3.75e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --dataset coco_base
