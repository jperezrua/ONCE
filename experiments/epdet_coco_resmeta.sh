# python main_ep.py epdet --exp_id coco_ep_resms18 --dataset coco_ep --arch resmsrw_18 --batch_size 16 --master_batch 8 --lr 1.00e-4 --gpus 0,1 --num_workers 16 --load_basemodel '../models/coco_base_res18_nohc/model_best.pth' --load_metamodel '../models/coco_base_res18_nohc/model_best.pth' --wh_weight 0.000001  --off_weight 0.00001 --head_conv 0
# EPISODIC TRAINING 
################ COCO EP RES-18 (RES-18 is used for both meta and base models)
python main_ep.py epdet --exp_id coco_ep_resmeta18 --dataset coco_ep --arch resmeta_18 --batch_size 40 --master_batch 10 --lr 4.00e-4 --gpus 0,1,2,3 --num_workers 16 --load_basemodel '../models/coco_res18_nohc/model_best.pth' --load_metamodel '../models/coco_res18_nohc/model_best.pth' --head_conv 0
python main_ep.py epdet --exp_id coco_ep_resmeta18 --dataset coco_ep --arch resmeta_18 --batch_size 40 --master_batch 10 --lr 1.00e-4 --gpus 0,1,2,3 --num_workers 16 --load_basemodel '../models/coco_res18_nohc/model_best.pth' --load_metamodel '../models/coco_res18_nohc/model_best.pth' --head_conv 0
python main_ep.py epdet --exp_id coco_ep_resmeta18 --dataset coco_ep --arch resmeta_18 --batch_size 10 --master_batch 10 --lr 1.00e-4 --gpus 0 --num_workers 16 --load_basemodel '../models/coco_res18_nohc/model_best.pth' --load_metamodel '../models/coco_res18_nohc/model_best.pth' --head_conv 0

#python main_ep.py epdet --exp_id coco_ep_resmetafull50 --dataset coco_ep --arch resmetafull_50 --batch_size 24 --master_batch 6 --lr 1.00e-4 --gpus 0,1,2,3 --num_workers 16 --head_conv 0 --n_class 3 --k_shots 5 --cat_spec_wh --load_basemodel ../models/coco_res50_nohc_few/model_last.pth  --load_metamodel ../models/coco_res50_nohc_few/model_last.pth 

################ COCO EP RES-50 (RES-50 is used for both meta and base models)
python main_ep.py epdet --exp_id coco_ep_resmeta50 --dataset coco_ep --arch resmeta_50 --batch_size 80 --master_batch 10 --lr 1.00e-5 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --load_basemodel '../models/coco_res50_nohc/model_best.pth' --load_metamodel '../models/coco_res18_nohc/model_best.pth' --head_conv 0

################# COCO FULL RES-101 (RES-18 is used for meta)
python main.py epdet --exp_id coco_ep_resmeta101  --dataset coco_ep --arch resmeta_101 --batch_size 80 --master_batch 10 --lr 1.00e-5 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --head_conv 0 --load_basemodel '../models/coco_res101_nohc/model_best.pth' --load_metamodel '../models/coco_res18_nohc/model_best.pth' --head_conv 0

