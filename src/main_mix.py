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
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model for support...')
  model_heads = {'hm': 1,
                'wh': 2,
                'reg': 2}
  model_supp = create_model(opt.arch_supp, model_heads, 
                opt.head_conv, extras={'learnable': opt.learnable, 'metasize': opt.metasize})

  if opt.load_suppmodel:
    print('=====> Loading pretrained models for support')
    l = torch.load(opt.load_suppmodel)['state_dict']
    model_supp.load_state_dict(l, strict=True)
    
  else:
    print('Cannot continue without a valid support model')
    exit()

  model_heads = {'hm': opt.num_classes,
                'wh': 2,
                'reg': 2}

  model = create_model(opt.arch, model_heads, 
                  opt.head_conv, extras={'learnable': opt.learnable, 'metasize': opt.metasize})
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)

  start_epoch = 0
  
  Trainer = train_factory['ctdet']
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )


  print('Extracting weights from supp model given support samples...')
  support_set = train_loader.dataset.get_support_set()
  support_set.to(opt.device)
  support_code = model_supp.rw.extract_support_code(support_set)
  print(support_code.shape)
  print(model.hm.weight.shape)
  print(model.wh.weight.shape)
  print('Initializing corresponding weights of model')
  # recall the weights need to be indexed. indexes of base classes influence output for base classes.


  print('Load weights for model from episodic training')
  mk,_ = model.load_state_dict( model_supp.state_dict(), strict=False )
  print('Missing Keys for model:    ',mk)
  
  print('Initialize reg weights')
  model.reg.weight.data = model_supp.reg.weight.data
  model.hm.weight.data[train_loader.dataset._simple_novel_ids] = support_code[:,:256].view(20,256,1,1).to(opt.device)
  model.wh.weight.data = torch.mean(support_code[:,256:],dim=0).view(2,256,1,1).to(opt.device)
  model.to(opt.device)

  model_supp = model_supp.cpu()

  print('Starting training...')
  best = 1e10
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
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