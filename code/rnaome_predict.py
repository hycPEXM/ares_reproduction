import argparse as ap
import logging
import os
import sys

import atom3d.datasets as da
import dotenv as de
import pandas as pd
import pathlib
import pytorch_lightning as pl
import torch_geometric

import ares.data as d
import ares.rnaome_model as m

root_dir = pathlib.Path(__file__).parent.parent.absolute()
de.load_dotenv(os.path.join(root_dir, '.env'))
logger = logging.getLogger("lightning")

'''
p = pathlib.PurePosixPath('/a/b/c/d')
print(p,p.parent,p.parent.parent)
root_dir = pathlib.Path(__file__).parent.absolute()
print(root_dir)
print(pathlib.Path(__file__).parent.parent.absolute(),os.path.join(pathlib.Path(__file__).parent.parent.absolute(), '.env'))
print(os.path.join(root_dir, '.env'))

# 输出结果：
/a/b/c/d /a/b/c /a/b
/fsa/home/ww_duyy/hyc/data/ares_release/ares_release/ares
/fsa/home/ww_duyy/hyc/data/ares_release/ares_release /fsa/home/ww_duyy/hyc/data/ares_release/ares_release/.env
/fsa/home/ww_duyy/hyc/data/ares_release/ares_release/ares/.env
#说明原来predict.py里root_dir的定义是错的

'''

def main():
    parser = ap.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('dataset', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('-f', '--filetype', type=str, default='lmdb',
                        choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--label_dir', type=str, default=None)
    parser.add_argument('--nolabels', dest='use_labels', action='store_false')
    parser.add_argument('--num_workers', type=int, default=1)

    # add model specific args
    parser = m.ARESModel.add_model_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dict_args = vars(hparams)

    transform = d.create_transform(
        hparams.use_labels, hparams.label_dir, hparams.filetype)

    # DATA PREP
    logger.info(f"Dataset of type {hparams.filetype}")
    dataset = da.load_dataset(hparams.dataset, hparams.filetype, transform)
    dataloader = torch_geometric.data.DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)

    # MODEL SETUP
    logger.info("Loading model weights...")
    tfnn = m.ARESModel.load_from_checkpoint(hparams.checkpoint_path)

    trainer = pl.Trainer.from_argparse_args(hparams)

    # PREDICTION
    logger.info("Running prediction...")
    out = trainer.test(tfnn, dataloader, verbose=False)

    # SAVE OUTPUT
    df = pd.DataFrame(tfnn.predictions)
    for index,row in df.iterrows():
        df.loc[index, 'tag']=str(row['tag'])[:-4]   #去掉.pdb后缀；也可尝试apply方法
    df.to_csv(hparams.output_file, index=False, float_format='%.7f')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    main()
