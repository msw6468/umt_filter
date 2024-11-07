# Ignore warning
# import warnings
# warnings.filterwarnings(action='ignore')
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
import h5py
from pathlib import Path
from abc import ABC
import io

import numpy as np
import pandas as pd

import torch
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import transforms, InterpolationMode

from PIL import Image

import argparse
import colorful
from tqdm import tqdm

from pprint import pprint

# For UMT
from tasks.shared_utils import setup_model
from utils.config_utils import setup_main
from models.umt import UMT
from models.backbones.bert.tokenization_bert import BertTokenizer
from einops import rearrange

LOAD_DIR = {
    'millet': '/gallery_millet/chris.kim/data/webvid',
    'tate':   '/gallery_tate/dongyeon.woo/jongchan/webvid',
    'getty':  '/gallery_getty/dongjoo.kim/vision/webvid',
    'moma':   '/gallery_moma/jongchan.noh/webvid',
    'orsay':  '/gallery_orsay/sangwoo.moon/JC/webvid',
}

# utils ----------------------------------------------------------
def get_UMT_model(args):
    """ Get UMT models
    Return:
        - config
        - model
        - transform
        - tokenizer
    """
    config                              = setup_main(args)
    config.scheduler.num_training_steps = 0
    config.scheduler.num_warmup_steps   = 0
    model_cls                           = eval(config.model.get('model_cls', 'UMT'))

    (model, model_without_ddp, optimizer, scheduler, scaler,
        tokenizer, start_epoch, global_step,
    ) = setup_model(
        config, model_cls=model_cls, has_decoder=False,
        pretrain=False, find_unused_parameters=False,
    )

    # sanity check
    # with torch.cuda.amp.autocast(enabled=config.fp16):
    #     model.eval()
    #     image = torch.ones(32,4,3,224,224).to(args.device)
    #     use_image=False
    #     keep_temporal=True
    #     model.encode_vision(image, NOne, use_image, keep_temporal)
    #     from IPython import embed; embed(colors='neutral')  # XXX DEBUG  # yapf: disable

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Set transform
    normalize = transforms.Normalize(mean, std)
    type_transform = transforms.Lambda(lambda x: x.float().div(255.0))

    transform = transforms.Compose(
        [transforms.Resize((config.inputs.image_res, config.inputs.image_res),
                            interpolation=InterpolationMode.BICUBIC,),
        type_transform,
        normalize,])

    return config, model, tokenizer, transform


def get_total_video_dict(args):
    file_name = f'{LOAD_DIR["millet"]}/webvid10m_train_{args.dir_name}_w_uid_part{args.meta_part}.csv'
    total_video_dict = pd.read_csv(file_name, index_col=0)
    return total_video_dict


def get_partitioned_dict(total_video_dict, total, part):
    if total==1:
        return total_video_dict

    total_size       = len(total_video_dict)
    part_size        = int(total_size / total)

    start            = part_size * (part - 1)
    end              = part_size * (part) if part < total else total_size

    new_total_video_dict  = total_video_dict.iloc[start:end]
    print(f'[PARTITION] Total datasets  : {len(total_video_dict)}, Part: {part}/{total} [{start}:{end}]')
    print(f'[PARTITION] After partition : {len(new_total_video_dict)}')

    return new_total_video_dict


def get_preprocessed_frames_hdf5(args):
    h5_filename = os.path.join(LOAD_DIR[args.dir_name], f'preprocessed_frames_{args.dir_name}_{args.meta_part}.h5')
    h5_file = h5py.File(h5_filename, 'r')
    return h5_file


def get_h5py_files(args):
    h5py_f = {}

    # h5py_f['text_feats_h5']  = h5py.File(os.path.join(args.save_path, f'text_feats_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    # h5py_f['text_atts_h5']   = h5py.File(os.path.join(args.save_path, f'text_atts_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    # h5py_f['image_feats_h5'] = h5py.File(os.path.join(args.save_path, f'image_feats_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    h5py_f['clip_sim_h5']    = h5py.File(os.path.join(args.save_path, f'clip_sim_part_{args.dir_name}_{args.meta_part}_{args.total}_{args.part}.h5'), 'a')

    for key in h5py_f:
        h5py_f[key].flush()
        os.chmod(h5py_f[key].filename, mode=0o777)

    return h5py_f

# ----------------------------------------------------------------

# ================
# Generic Datasets
# ================
class BaseDataset(Dataset, ABC):
    name = 'base'
    dataset_size = 0

    def __init__(self):
        super().__init__()

    def __len__(self):
        return self.dataset_size

    def collate_fn(self, batch):
        return default_collate(batch)
# ================
# WebVid  Datasets
# ================
class WebVid(BaseDataset):
    name = 'webvid'
    def __init__(self, args, tokenizer, processor, frames_h5=None, video_dict=None):
        super(WebVid, self).__init__()

        self.args   = args
        self.debug  = args.debug
        self.device = args.device

        self.max_frames = args.max_frames
        self.frame_idxs = [0,1,2,3,4,5,6,7] if self.max_frames==8 else [0,2,4,6] # UMT case
        self.max_words  = 77

        self.frames_h5 = frames_h5

        # Set preprocess
        self.tokenizer = tokenizer
        self.processor = processor

        # Set dataframe
        self.df = video_dict


    def _get_frames(self, video_id=None, text_id=None):
        """ Get video information
        INPUT:
            video_id: video_id
            text_id : text_id
        OUTPUT:
            clip_images: (3, max_frame, 224, 224), torch.tensor
                - max_frame : max frame per clip
        """
        #====================
        # Get all frames
        #====================
        if self.args.frame_load == 'hdf5':
            try:
                # Load from frame -----------------------------------------
                binary_images = self.frames_h5[video_id][self.frame_idxs]
                images = torch.zeros((self.max_frames, 224, 224, 3))

                for i, binary_image in enumerate(binary_images):
                    images[i] = torch.from_numpy(np.array(Image.open(io.BytesIO(binary_image))))

                # images = images.permute(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W) # LB
                images = images.permute(0, 3, 1, 2) # (T, H, W, C) -> (T, C, H, W)   # UMT
                images = self.processor(images)
                # ----------------------------------------------------------

            except Exception as e:
                print(f'video_id {video_id}, text_id {text_id} sample is corrupted, {e}')
                return torch.zeros((3, self.max_frames, 224, 224), dtype=torch.float), False # bad clip-captions
        else:
            NotImplementedError

        return images, True


    def __repr__(self):
        return str(self)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data               = self.df.iloc[idx]
        unique_id          = data['unique_id'] # use idx as unique identifier (name for dataset in h5 file)
        video_id           = f"{data['page_dir']}/{data['videoid']}" 
        raw_text           = data['name']
        frames, valid_flag = self._get_frames(video_id=video_id)

        return unique_id, frames, raw_text, valid_flag


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    # For partition
    parser.add_argument("--dir_name",  type=str, default='moma', help="[moma, tate, getty, orsay, ai2]")
    parser.add_argument("--meta_part", type=int, default=1,      help="after multi-downloading. which part?(0, ..., )")

    parser.add_argument("--part",      type=int, default=1,      help="for mulit_running. which part?(1, ..., total)")
    parser.add_argument("--total",     type=int, default=1,      help="for multi_running. how many parts?")

    # For dataloader
    parser.add_argument("--device",      type=str, default='cuda')
    parser.add_argument("--batch_size",  type=int, default=200, help="[100 for 24GB, 200 for 48GB]")
    parser.add_argument("--num_workers", type=int, default=4)

    # others
    parser.add_argument("--num_segment",  type=int, default=50)

    parser.add_argument("--debug",      type=str, default='False')
    parser.add_argument("--frame_load", type=str, default='hdf5', help='[image, hdf5]')
    parser.add_argument("--max_frames", type=int, default=4, help='[4,  8]')
    parser.add_argument("--final_check", type=str, default='False', help='[True, False]')

    parser.add_argument("--config_file", type=str, default='multi_modality/exp/zero_shot/ret_msrvtt/l16.py')
    parser.add_argument("--pretrained_path", type=str, default='multi_modality/pretrained_model/l16_5m.pth')
    parser.add_argument("--output_dir", type=str, default='multi_modality/checkpoints')

    return parser.parse_args()

# %%
def main(args):
    args.debug = True if args.debug == 'True' else False
    args.final_check = True if args.final_check == 'True' else False
    args.root_path = os.path.join(LOAD_DIR['orsay'])
    args.save_path = os.path.join(args.root_path, 'UMT') if not args.debug else os.path.join(args.root_path, 'UMT', 'debug')

    pprint(args)
    print(f'Debug mode: {args.debug}')

    print('Load model')
    config, model, tokenizer, transform = get_UMT_model(args)
    text_encoder = model.get_text_encoder()

    print('Load json(total_video_dict)')
    total_video_dict = get_total_video_dict(args)
    print(f'Load json(total_video_dict) results: Number of videos = {len(total_video_dict)}')

    print(f'[PARTITION] Select data {args.part}/{args.total}')
    total_video_dict = get_partitioned_dict(total_video_dict, args.total, args.part)

    print(f'Set save path on {args.save_path}')
    args.flag_dir   = os.path.join(args.save_path, 'final_flag')
    os.makedirs(args.flag_dir, exist_ok=True, mode=0o777)
    h5py_f = get_h5py_files(args)

    print(f'Load preprocessed_frames')
    frames_h5 = get_preprocessed_frames_hdf5(args)

    # remove processed indexs
    processed_index_list = os.listdir(args.flag_dir)
    processed_index_list = list(map(int, processed_index_list))
    total_video_dict   = total_video_dict[~total_video_dict['unique_id'].isin(processed_index_list)]
    print(f'After remove processed indexes : {len(total_video_dict)}')

    dataset  = WebVid(
        args       = args,
        tokenizer  = tokenizer,
        processor  = transform,
        video_dict = total_video_dict,
        frames_h5  = frames_h5,)

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        # collate_fn  = dataset.collate_fn,
        drop_last   = False,
        pin_memory  = True,  # better when training on GPU.
        shuffle     = False) # Don't need to shuffle for captioning

    print('Start batch')
    # text_feats_dict = {}
    # text_atts_dict = {}
    # image_feats_dict = {}
    clip_sim_dict = {}
    step = 0

    with torch.no_grad():
        model.eval()
        with torch.cuda.amp.autocast(enabled=config.fp16):
            pbar = tqdm(dataloader)
            for batch in pbar:
                pbar.set_description(f"[{args.dir_name}_{args.meta_part}[{args.part:2d}/{args.total:2d}]")
                unique_ids, frames, raw_texts, valid_flag = batch
                if np.sum(np.array(valid_flag)) == 0:
                    continue
                unique_ids = np.array(unique_ids)[np.array(valid_flag)]

                # retrieval_utils.py extract_text_feats() L18
                raw_texts  = list(np.array(raw_texts)[np.array(valid_flag)])
                texts      = dataset.tokenizer(raw_texts,
                                            max_length=dataset.max_words,
                                            padding='max_length',
                                            truncation=True,
                                            return_tensors='pt').to(args.device)
                text_feats = model.encode_text(texts)[0]
                text_atts  = texts['attention_mask']
                # -------------------------------------------------------------

                # retrieval_utils.py extract_vision_feats() L43
                frames         = frames[valid_flag].to(args.device)
                image_feats, _ = model.encode_vision(frames, test=True)

                if config.evaluation.eval_frame_ensemble == "concat":  # default
                    if len(image_feats.shape) == 4:
                        image_feats = rearrange(image_feats, "b t l c -> b (t l) c").contiguous()
                    image_feats = image_feats  # (bsz, 1, #frm*L, d)
                else:
                    assert config.video_input.num_frames == 1, "only support single-frame"
                    assert config.evaluation.eval_frame_ensemble in ["mean", "max", "lse"]
                # -------------------------------------------------------------

                encoder_output = image_feats # (#frm*Li, d)
                encoder_att    = torch.ones(encoder_output.size()[:-1],
                                        dtype=torch.long).to(args.device, non_blocking=True)

                output = text_encoder(
                    encoder_embeds=text_feats,
                    attention_mask=text_atts,
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_att,
                    return_dict=True,
                    mode="fusion",
                )

                itm_embeds = output.last_hidden_state[:, 0]
                similarity = model.itm_head(itm_embeds)[:, 1]

                similarity  = similarity.unsqueeze(1).detach().cpu().numpy()
                # text_feats  = text_feats.unsqueeze(1).detach().cpu().numpy()
                # text_atts   = text_atts.unsqueeze(1).detach().cpu().numpy()
                # image_feats = image_feats.unsqueeze(1).detach().cpu().numpy()  # (bsz, 1, #frm*L, d)

                for u_id, sim in zip(unique_ids, similarity):
                    if str(u_id) in h5py_f['clip_sim_h5'].keys():
                        del h5py_f['clip_sim_h5'][str(u_id)]

                    h5py_f['clip_sim_h5'].create_dataset(str(u_id), data = np.array([[sim]]))
                    h5py_f['clip_sim_h5'].flush()
                    flag_save_path = os.path.join(args.flag_dir, f'{u_id}')
                    Path(flag_save_path).touch()

                step += 1

                if step == args.num_segment:
                    for key in h5py_f.keys():
                        h5py_f[key].flush()

    for key in h5py_f.keys():
        h5py_f[key].close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print(colorful.bold_pink("Thank you and Good Job Computer.").styled_string)
