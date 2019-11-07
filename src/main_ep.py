from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory


def main(opt):
  #torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

  if opt.task != 'reweight_paper':
    Dataset = get_dataset(opt.dataset, opt.task)
  else:
    #python main_ep.py reweight_paper --load_model ../models/best_resmeta_50/model_last.pth  --fewshot_data basenovel_fewshot --master_batch_size 1 --batch_size 8 --head_conv 0 --cat_spec_wh --lr 1e-5 --gpus 0,1,2,3,4,5,6,7,8
    Dataset = get_dataset('coco_mixed', 'mixdet')

  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  if opt.task != 'reweight_paper':
    model = create_model(opt.arch, opt.heads, opt.head_conv, extras={'learnable': opt.learnable, 'metasize': opt.metasize})
  else:
    print(100*'=')
    print(opt.heads, opt.head_conv)
    print(Dataset.num_classes)
    model = create_model('resmeta_50', opt.heads, opt.head_conv, extras={'learnable': opt.learnable, 'metasize': opt.metasize})

  if opt.task != 'reweight_paper':
    optimizer = torch.optim.Adam(model.meta_params, opt.lr)
  else:
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

  start_epoch = 0
  if opt.task != 'reweight_paper':
    if opt.load_model != '':
      model, optimizer, start_epoch = load_model(
        model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
  else:
    if opt.load_model != '':
      l = torch.load(opt.load_model)['state_dict']    
      mk,uk = model.load_state_dict(l, strict=False)
      print('Missing Keys for model:    ',mk)
      print('')
      print('Unknown Keys for model:    ',uk)      

  if opt.task != 'reweight_paper':
    Trainer = train_factory[opt.task]
  else:
    Trainer = train_factory['epdet']

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  if opt.task != 'reweight_paper':
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val', base=True), #the 20 left-out classes are only for testing
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train', base=True), 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
  else:
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )    

    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'), 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

  if opt.task != 'reweight_paper':
    if opt.load_basemodel:
      print('=====> Loading pretrained models')
      l = torch.load(opt.load_basemodel)['state_dict']
      mk,uk = model.load_state_dict(l, strict=False)
      print('Missing Keys for base model:    ',mk)
      print('')
      print('Unknown Keys for base model:    ',uk)

    if opt.load_metamodel:
      print('=====> Loading meta-pretrained models')
      l = torch.load(opt.load_metamodel)['state_dict']
      mk,uk = model.rw.load_state_dict(l, strict=False)
      print('Missing Keys for meta model:    ',mk)
      print('')
      print('Unknown Keys for meta model:    ',uk)

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):

    # hardcoded warmpup
    if opt.task != 'reweight_paper':
      if epoch == 1:
        lr = opt.lr/100
        print('**** Warmup LR to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
      elif epoch == 2:
        lr = opt.lr
        print('**** Warmup finished to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if log_dict_val[opt.metric] < best:
        best = log_dict_val[opt.metric]
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model)
    else:
      save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                 epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)