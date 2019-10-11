
# EPISODIC TRAINING 
################ COCO EP RES-18 (RES-18 is used for both meta and base models)
python main_ep.py epdet --exp_id coco_ep_res18 --dataset coco_ep --arch resrw_18 --batch_size 96 --master_batch 18 --lr 4.00e-4 --gpus 0,1,2,3 --num_workers 16 --load_basemodel '../models/coco_base_res18_nohc/model_best.pth' --load_metamodel '../models/coco_base_res18_nohc/model_best.pth' --wh_weight 0.000001  --off_weight 0.00001 --head_conv 0



################# COCO FULL RES-101 (RES-18 is used for meta)
python main.py epdet --exp_id coco_ep_res101_nohc  --dataset coco_ep --arch resrw_101 --batch_size 84 --master_batch 5 --lr 3.00e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --head_conv 0 --load_basemodel '../models/coco_base_res101_nohc/model_best.pth' --load_metamodel '../models/coco_base_res18_nohc/model_best.pth' --wh_weight 0.000001  --off_weight 0.00001 --head_conv 0
