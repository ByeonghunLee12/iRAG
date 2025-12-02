import math
import random
import argparse
from collections import OrderedDict
from datetime import timedelta
from timeit import default_timer as timer

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.logger import Logger
from utils.data import ImageFolderData
import os 
import torch.nn.functional as F 
import time


_WARNING_DOCS = f'[W] Please ensure the database size is divisible by the batch size, so the model can find the optimal image pairs.\n\
[W] Using CLIP shows better qualitative results, so if VGG Baseline retrieval is not good enough, \
try using CLIP by setting --use_clip True'


class Base_Model(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.register_data()
        self.register_device()
        self.register_save_path()

    def register_save_path(self):
        self.save_root = self.hparams.database_path.split('/')[-1] + '_results'
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
            os.makedirs(f'{self.save_root}/txt')
            os.makedirs(f'{self.save_root}/models')
        else: 
            self.save_root = self.save_root + f'_{time.time()}'
            self.save_root = self.save_root[:self.save_root.rfind('.')]
            print(f'[A] Results will be saved to {self.save_root}\n')
            os.makedirs(self.save_root)
            os.makedirs(f'{self.save_root}/txt')
            os.makedirs(f'{self.save_root}/models')

    def register_data(self):
        self.database_path = self.hparams.database_path  # equivalent to train data
        self.query_path    = self.hparams.query_path     # equivalent to validation data

        self.db_data       = ImageFolderData()
        self.data          = ImageFolderData()

    def register_device(self):
        self.device = torch.device('cuda')

    def train_step(self):
        logger = Logger(self.save_root + '/log.log', on=True)
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.single_train_step(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.4f}, saving'.format(val_perf))
                torch.save({
                    'hparams': self.hparams,
                    'state_dict': state_dict
                }, self.hparams.model_path)

        logger.log('Time: {}'.format(timedelta(seconds=round(timer() - start))))

    def single_train_step(self, run_num, logger):
        self.train()

        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: {}/{}'.format(run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                setattr(self.hparams, hparam, random.choice(values))

        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        self.define_parameters()

        self.hparams.epochs = max(1, self.hparams.epochs)

        logger.log(f'\n[Hyperparameters for run]')
        for hparam, val in vars(self.hparams).items():
            logger.log(f'{hparam}: {val}')

        self.to(self.device)

        optimizer = self.configure_optimizers()

        # [1] Train 
        train_db_loader = self.db_data.get_loader(
            img_dir=self.database_path,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            mode='train', drop_last = True,
            strict_lr_hr_correlation=self.hparams.enf_corr, 
        )

        # [2] Inference 
        test_db_loader = self.data.get_loader(
            img_dir=self.database_path,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            mode='test'
        )
        test_query_loader = self.data.get_loader(
            img_dir=self.query_path,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            mode='test'
        )

        best_val_perf = float('-inf')
        best_state_dict = None
        try:
            for epoch in range(1, self.hparams.epochs + 1):
                forward_sum = {}
                num_steps = 0

                for batch in tqdm(train_db_loader, desc='Training Batches', total=len(train_db_loader)):
                    optimizer.zero_grad()
                    imgi, imgj, _ = batch
                    imgi = imgi.to(self.device)
                    imgj = imgj.to(self.device)

                    forward = self.forward(imgi, imgj)

                    # Sum up losses and metrics.
                    for key in forward:
                        forward_sum[key] = forward_sum.get(key, 0) + forward[key]
                    num_steps += 1

                    if math.isnan(forward_sum.get('loss', 0)):
                        logger.log('Stopping epoch because loss is NaN')
                        break

                    forward['loss'].backward()
                    optimizer.step()

                if math.isnan(forward_sum.get('loss', 0)):
                    logger.log('Stopping training session because loss is NaN')
                    break

                logger.log('End of epoch {:3d}'.format(epoch))
                logger.log(' '.join(['| {} {:8.4f}'.format(key, forward_sum[key] / num_steps)
                                     for key in forward_sum]))

                if epoch % self.hparams.validate_frequency == 0:
                    print('==== Evaluating via Ordering ====')
                    self.register_retrieval_fn('Rank:1')
                    retrieved = self.retrieval(test_db_loader, test_query_loader, topK=5)

                    with open(f'{self.save_root}/txt/results_{epoch}_train_down.txt', 'w') as f:
                        f.write('\n'.join(
                            f"{qp}\t{', '.join(sp)}" for qp, sp in retrieved
                        ) + '\n')
                    # Save current state for backup.
                    torch.save({
                        'hparams': self.hparams,
                        'state_dict': self.state_dict()
                    }, f'{self.save_root}/models/{epoch}_train_down.pth')

        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')

        return best_state_dict, best_val_perf

    def run_evaluation_sessions(self): 
        logger = Logger(self.save_root + '/log.log', on=True)
        random.seed(self.hparams.seed)
        retrieval_fn = 'Rank:1'
        self.register_retrieval_fn(retrieval_fn)
        self.run_evaluation_session(retrieval_fn, logger)

    def register_retrieval_fn(self, retrieval_fn: str='Rank:1'): 
        if retrieval_fn == 'BASIC': 
            retrieval_config = {
                'ordering': False, 
                'reverse': False
            }
        elif retrieval_fn == 'INV': 
            retrieval_config = {
                'ordering': False, 
                'reverse': True
            }
        elif retrieval_fn == 'Rank:1': 
            retrieval_config = {
                'ordering': True, 
                'reverse': False
            }
        elif retrieval_fn == 'Rank:-1': 
            retrieval_config = {
                'ordering': True, 
                'reverse': True
            }
        else: 
            raise NotImplementedError
        self.retrieval_config = retrieval_config

    def run_evaluation_session(self, retrieval_fn: str='Rank:1', logger=None):
        self.eval()
        self.to(self.device)

        # Log the start of evaluation.
        logger.log("Starting evaluation session.")

        # [1] Set DATALOADER 
        test_db_loader = self.data.get_loader(
            img_dir=self.database_path,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            mode='test'
        )
        test_query_loader = self.data.get_loader(
            img_dir=self.query_path,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            mode='test'
        )

        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        logger.log(f"Running retrieval evaluation... with {self.retrieval_config}")
        retrieved = self.retrieval(test_db_loader, test_query_loader, topK=5)

        # mkdir
        if not os.path.exists(f'{self.save_root}_eval/txt'):
            os.makedirs(f'{self.save_root}_eval/txt')
        results_path = f'{self.save_root}_eval/txt/results_eval.txt'
        with open(results_path, 'w') as f:
            f.write('\n'.join(
                f"{qp}\t{', '.join(sp)}" for qp, sp in retrieved
            ) + '\n')

        logger.log(f"Evaluation complete. Results saved to: {results_path}")

        return retrieved      

    def retrieval(self, 
                  database_loader, 
                  val_loader, 
                  topK):
        """
        Retrieve the top-K similar images for each query image.
        """
        retrieval_cfgs = self.retrieval_config
        _encode_fn     = self._encode_loader_continuous if retrieval_cfgs['ordering'] else self._encode_loader

        def _get_similarity(queryB: torch.Tensor, 
                            retrievalB: torch.Tensor) \
                                -> torch.Tensor: 
            if retrieval_cfgs['ordering']:
                retrievalN = F.normalize(retrievalB.float(), p=2, dim=1)
                queryN     = F.normalize(queryB.float(),     p=2, dim=1)
                return torch.mm(queryN, retrievalN.t())
            else: 
                return torch.mm(queryB.float(), retrievalB.float().t())

        self.eval()
        with torch.no_grad():
            retrievalB, retrievalPaths = _encode_fn(database_loader)
            queryB, queryPaths         = _encode_fn(val_loader)

            similarity         = _get_similarity(queryB, retrievalB)
            _, top_indices     = torch.topk(similarity, k=topK, dim=1, largest=retrieval_cfgs['reverse']==False, sorted=True)

            results = [
                (queryPaths[i], [retrievalPaths[idx] for idx in top_indices[i].cpu().tolist()])
                for i in range(top_indices.size(0))
            ]

        self.train()
        return results

    def _encode_loader(self, loader):
        """
        Encode all images in a DataLoader.
        """
        codes = []
        paths = []
        for data, p in loader:
            data = data.to(self.device, non_blocking=True)
            codes.append(self.encode_discrete(data).to(torch.int8))
            paths.extend(p)
        return torch.cat(codes, dim=0), paths

    def _encode_loader_continuous(self, loader):
        """
        Encode all images in a DataLoader.
        """
        codes = []
        paths = []
        # tqdm 
        for data, p in tqdm(loader, desc='Encoding Batches', total=len(loader)):
            data = data.to(self.device, non_blocking=True)
            codes.append(self.encode_continuous(data))
            paths.extend(p)
        return torch.cat(codes, dim=0), paths

    def flag_hparams(self):
        flags = self.hparams.model_path
        for hparam, val in vars(self.hparams).items():
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --{}'.format(hparam)
            elif hparam in {'model_path', 'num_runs', 'num_workers'}:
                continue
            else:
                flags += ' --{} {}'.format(hparam, val)
        return flags

    def load(self, model_path: str='path.pth'):
        device = torch.device('cuda')
        ckpt   = torch.load(model_path)

        self.hparams = ckpt['hparams']
        self.define_parameters()
        self.load_state_dict(ckpt['state_dict'])
        self.to(device)

    @staticmethod
    def get_general_hparams_grid():
        return OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.003, 0.001, 0.0003, 0.0001],
            'batch_size': [64, 128, 256],
        })

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--train', action='store_true', help='Train a model?')
        parser.add_argument('--query_path', type=str, default='path/to/query/images',
                            help='Path to the query images')
        parser.add_argument('--database_path', type=str, default='path/to/database/images',
                            help='Path to the database images')
        parser.add_argument("-l", "--encode_length", type=int, default=16,
                            help="Number of bits of the hash code [default: 16]")
        parser.add_argument("--lr", default=1e-3, type=float,
                            help='Initial learning rate [default: 1e-3]')
        parser.add_argument("--batch_size", default=64, type=int,
                            help='Batch size [default: 64]')
        parser.add_argument("-e", "--epochs", default=60, type=int,
                            help='Max number of epochs [default: 60]')

        parser.add_argument('--use_clip', action='store_true', help='Use CLipIBHash?', default=False)

        parser.add_argument('--num_runs', type=int, default=1,
                            help='Number of random runs [default: 1]')
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='Number of indulged bad epochs [default: 6]')
        parser.add_argument('--validate_frequency', type=int, default=20,
                            help='Validate every N epochs [default: 20]')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='Number of dataloader workers [default: 1]')
        parser.add_argument('--seed', type=int, default=8888,
                            help='Random seed [default: 8888]')
        parser.add_argument('--device', type=int, default=0, help='GPU device ID')
        parser.add_argument('--ckpt_path', 
                            default=None, 
                            type=str, help='Path to the pretrained model')


        parser.add_argument('--enf_corr', action='store_true', help='Enforce LR-HR correlation?', default=False)
        print(_WARNING_DOCS)

        return parser

    # Mother function
    def get_hparams_grid(self):
        raise NotImplementedError

    def define_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
