import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from model.CIBHash import CIBHash

if __name__ == '__main__':
    argparser = CIBHash.get_model_specific_argparser()
    hparams = argparser.parse_args()
    torch.cuda.set_device(hparams.device)

    if hparams.use_clip:
        from model.CLipIBHash import CLipIBHash
        print(f'[A] Using CLIP model for the encoder')
        model = CLipIBHash(hparams)
    else:
        print(f'[A] Using baseline (VGG16) model for the encoder')
        model = CIBHash(hparams)

    device = torch.device('cuda')
    model.to(device)
    if hparams.train:
        model.module.train_step() if isinstance(model, nn.DataParallel) else model.train_step()
    else:
        model_path = hparams.ckpt_path
        model.load(model_path)
        print(f'Loaded model from {model_path.split("/")[-1]}')
        model.run_evaluation_sessions()
