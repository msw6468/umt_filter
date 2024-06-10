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
    'ai2'    : '/net/nfs3.prior/dongjook/',
    'tate'    : '/gallery_tate/dongyeon.woo/howto100m/', # /8frames_per_clip/preprocessed_frames_metapart15_part0.h5py
    'moma'    : '/gallery_moma/sangwoo.moon/data/video/howto100m/',
    'getty'    : '/gallery_getty/dongjoo.kim/vision/howto370k/',
    'millet'    : '/gallery_millet/chris.kim/data/howto100m/',}

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
    if args.dir_name == 'ai2':
        # e.g., /home/dongjook/data/nfs/730k/sentencified_htm_730k_part{4-9}}.json
        # e.g., /home/dongjook/data/nfs/subset/sentencified_htm_subset_part{0-3}.json
        file_name = f'sentencified_htm_{args.data_version}_part{args.meta_part}.json'
        with open(os.path.join(args.root_path, file_name), 'r') as f:
            total_video_dict = json.load(f)
    elif args.dir_name == 'align':
        with open(os.path.join(LOAD_DIR[args.dir_name], 'htm_align_reformatted.json'), 'r') as f:
            total_video_dict = json.load(f)
    else: # tate, moma, getty, orsay, millet, ...
        if args.data_version == '8frames_per_clip':
            with open(os.path.join(LOAD_DIR['tate'], 'sentencified', f'sentencified_htm_remaining_part{args.meta_part}.json'), 'r') as f:
                total_video_dict = json.load(f)
        else:
            with open(os.path.join(LOAD_DIR['tate'], 'sentencified', f'sentencified_htm_{args.data_version}_part{args.meta_part}.json'), 'r') as f:
                total_video_dict = json.load(f)

    return total_video_dict


def get_partitioned_dict(total_video_dict, total, part):
    if total==1:
        return total_video_dict

    if args.total > 1:
        total_video_list  = list(total_video_dict.keys())
        total_size        = len(total_video_dict)
        part_size         = int(total_size / total)

        start             = part_size * (part - 1)
        end               = part_size * (part) if part < total else total_size

        new_total_video_dict = {}
        total_video_list = total_video_list[start:end]
        for video_id in total_video_list:
            new_total_video_dict[video_id] = total_video_dict[video_id]
        print(f'[PARTITION] Total datasets  : {len(total_video_dict)}, Part: {part}/{total} [{start}:{end}]')
        print(f'[PARTITION] After partition : {len(new_total_video_dict)}')

    return new_total_video_dict


def get_preprocessed_frames_hdf5(args):
    if args.frame_load == 'hdf5':
        if args.dir_name == 'ai2':
            hdf5_file = h5py.File(os.path.join(args.root_path, f'preprocessed_frames_part{args.meta_part}.h5py'),'r')
        else: # our server
            if args.data_version == 'subset':
                hdf5_file = h5py.File(f'/gallery_moma/sangwoo.moon/data/video/howto100m_LB/subset/preprocessed_frames_part{args.meta_part}.h5py', 'r')
            elif args.data_version == 'valid':
                hdf5_file = h5py.File(f'/gallery_millet/chris.kim/data/howto100m/valid/preprocessed_frames_part{args.meta_part}.h5py', 'r')
            elif args.data_version == '8frames_per_clip':
                hdf5_file = h5py.File(os.path.join(args.root_path, f'preprocessed_frames_metapart{args.meta_part}_part{args.meta_sub_part}.h5py'),'r')
            else:
                hdf5_file = h5py.File(os.path.join(args.root_path, f'preprocessed_frames_metapart{args.meta_part}.h5py'),'r')
    else:
        NotImplementedError

    return hdf5_file


def get_h5py_files(args):
    h5py_f = {}

    h5py_f['text_ids_h5']    = h5py.File(os.path.join(args.save_path, f'text_ids_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    # h5py_f['text_feats_h5']  = h5py.File(os.path.join(args.save_path, f'text_feats_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    # h5py_f['text_atts_h5']   = h5py.File(os.path.join(args.save_path, f'text_atts_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    # h5py_f['image_feats_h5'] = h5py.File(os.path.join(args.save_path, f'image_feats_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')
    h5py_f['clip_sim_h5']    = h5py.File(os.path.join(args.save_path, f'clip_sim_part{args.meta_part}_{args.total}_{args.part}.h5'), 'a')

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
# HowTo100M  Datasets
# ================
class HowTo100M(BaseDataset):
    name = 'howto100m'
    def __init__(self, args, tokenizer, processor, valid_current_video_ids, frames_h5=None, video_dict=None):
        super(HowTo100M, self).__init__()

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

        self.video_dict = video_dict
        self.valid_current_video_ids = valid_current_video_ids

        # Set dataframe
        df = pd.DataFrame(columns = ['video_id', 'text_id'])
        for video_id in tqdm(valid_current_video_ids):
            max_text_idx       = len(self.video_dict[video_id]['text'])
            cur_df             = pd.DataFrame(columns = ['video_id', 'text_id'])
            cur_df['text_id']  = np.arange(0, max_text_idx)
            cur_df['video_id'] = video_id
            df = pd.concat([df, cur_df], axis=0, ignore_index=True)
        self.df = df


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
                binary_images = self.frames_h5[video_id][text_id*8: (text_id+1)*8][self.frame_idxs]
                images = torch.zeros((self.max_frames, 224, 224, 3))

                for i, binary_image in enumerate(binary_images):
                    images[i] = torch.from_numpy(np.array(Image.open(io.BytesIO(binary_image))))
                # Checking code!
                # Image.open(io.BytesIO(images[0])).save(f'temp/temp.jpg')

                # images = images.permute(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W) # LB
                images = images.permute(0, 3, 1, 2) # (T, H, W, C) -> (T, C, H, W)   # UMT
                images = self.processor(images)
                # ----------------------------------------------------------

            except Exception as e:
                print(f'video_id {video_id}, text_id {text_id} sample is corrupted, {e}')
                return torch.zeros((self.max_frames, 3, 224, 224), dtype=torch.float), False # bad clip-captions
        else:
            NotImplementedError

        return images, True


    def __repr__(self):
        return str(self)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        data               = self.df.iloc[idx]
        video_id           = data['video_id']
        text_id            = data['text_id']
        raw_text           = self.video_dict[video_id]['text'][text_id]
        frames, valid_flag = self._get_frames(video_id=video_id, text_id=text_id)

        return video_id, text_id, frames, raw_text, valid_flag

def get_UMT_scores(args, config, model,
                    text_feats,
                    text_atts,
                    image_feats,
                    ):

    device      = args.device
    text_feats  = torch.tensor(text_feats).to(device)
    text_atts   = torch.tensor(text_atts).to(device)
    image_feats = torch.tensor(image_feats).to(device)
    num_images  = len(image_feats)
    num_texts   = len(text_feats)

    # computes only part of the scores at each GPU, gather at the end
    print(f"Rerank dual-encoder results with cross-encoder...num_cilps: {num_images}")

    # XXX SW XXX ##############################################################
    # config.deep_fusion            : True
    # config.evaluation.eval_offload: True
    text_encoder = model.get_text_encoder()
    encoder_output = image_feats # (#frm*Li, d)
    encoder_att    = torch.ones(encoder_output.size()[:-1],
                            dtype=torch.long).to(device, non_blocking=True)
    itm_embeds     = []
    output = text_encoder(
        encoder_embeds=text_feats,
        attention_mask=text_atts,
        encoder_hidden_states=encoder_output,
        encoder_attention_mask=encoder_att,
        return_dict=True,
        mode="fusion",
    )
    itm_embeds = output.last_hidden_state[:, 0]
    score = model.itm_head(itm_embeds)[:, 1]
    i2t_scores_x = score.detach().cpu().numpy()

    # XXX SW XXX ##############################################################

    return i2t_scores_x

def save_embeds_sims_chunk_UMT_single(args, config, model,
                            text_ids_dict,
                            # text_feats_dict,
                            # text_atts_dict,
                            # image_feats_dict,
                            clip_sim_dict,
                            h5py_f,):
    # Save as single video id
    for video_id in clip_sim_dict.keys():
        # i2t_scores_x = get_UMT_scores(args, config, model,
        #             text_feats_dict[video_id],
        #             text_atts_dict[video_id],
        #             image_feats_dict[video_id],
        #             )

        h5py_f['text_ids_h5'].create_dataset(video_id, data = text_ids_dict[video_id])
        # h5py_f['text_feats_h5'].create_dataset(video_id, data = text_feats_dict[video_id])
        # h5py_f['text_atts_h5'].create_dataset(video_id, data = text_atts_dict[video_id])
        # h5py_f['image_feats_h5'].create_dataset(video_id, data = image_feats_dict[video_id])
        h5py_f['clip_sim_h5'].create_dataset(video_id, data = clip_sim_dict[video_id])

    # Flush
    for key in h5py_f:
        h5py_f[key].flush()

    # Save flag after flush
    if not args.debug:
        for video_id in clip_sim_dict.keys():
            flag_save_path = os.path.join(args.flag_dir, f'{video_id}')
            Path(flag_save_path).touch()
    else:
        from IPython import embed; embed(colors='neutral')  # XXX DEBUG  # yapf: disable
        pass

    return

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
    parser.add_argument("--data_version", type=str, required=True, help="[subset, 1200k, 730k, 370k, valid]")
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
    args.root_path = os.path.join(LOAD_DIR[args.dir_name], args.data_version)
    args.save_path = os.path.join(args.root_path, 'UMT') if not args.debug else os.path.join(args.root_path, 'UMT', 'debug')
    if (args.dir_name != 'ai2') and (args.data_version not in ['subset', 'valid']):
        args.meta_sub_part = (args.part-1) // 4
    pprint(args)
    print(f'Debug mode: {args.debug}')

    print('Load model')
    config, model, tokenizer, transform = get_UMT_model(args)
    text_encoder = model.get_text_encoder()

    print('Load json(total_video_dict)')
    total_video_dict = get_total_video_dict(args)
    print(f'Load json(total_video_dict) results: Number of videos = {len(total_video_dict)}')

    if hasattr(args, 'meta_sub_part'):
        print(f'[PARTITION] Select meta_sub_part\'s data {args.meta_sub_part+1}/4')
        total_video_dict = get_partitioned_dict(total_video_dict, 4, (args.meta_sub_part+1))
        total_video_ids = list(total_video_dict.keys())

        print(f'[PARTITION] Select data {args.part}/{args.total}')
        total_video_dict = get_partitioned_dict(total_video_dict, ((args.total-1)//4)+1, ((args.part-1)%4)+1)
        total_video_ids = list(total_video_dict.keys())

    else:
        print(f'[PARTITION] Select data {args.part}/{args.total}')
        total_video_dict = get_partitioned_dict(total_video_dict, args.total, args.part)
        total_video_ids = list(total_video_dict.keys())

    print(f'Set save path on {args.save_path}')
    args.flag_dir   = os.path.join(args.save_path, 'final_flag')
    os.makedirs(args.flag_dir, exist_ok=True, mode=0o777)
    h5py_f = get_h5py_files(args)

    print(f'Load preprocessed_frames')
    frames_h5 = get_preprocessed_frames_hdf5(args)

    for i in range(0, len(total_video_ids), args.num_segment):
        # Check current video ids and check its validity
        start = i
        end   = (i + args.num_segment) if (i + args.num_segment) < len(total_video_ids) else len(total_video_ids)
        current_video_ids = total_video_ids[start:end]

        # video validitiy check --------------------------------------------
        print(f'[Current_dataset] Validity check on video_id[{start}:{end}]')
        valid_current_video_ids = []
        for current_video_id in tqdm(current_video_ids):

            # check done_flag
            if args.final_check: # based on files
                current_video_id_done_flag = (current_video_id in h5py_f['clip_sim_h5'].keys())
            else:                # based on h5 file keys
                current_video_id_done_flag = os.path.exists(os.path.join(args.save_path, 'final_flag', current_video_id))

            # add video_id if not done
            if args.debug:
                valid_current_video_ids.append(current_video_id)
            else:
                if (not current_video_id_done_flag):
                    valid_current_video_ids.append(current_video_id)

        print(f'[Current_dataset] Validity check result: {len(valid_current_video_ids)}/{len(current_video_ids)}')

        if len(valid_current_video_ids) == 0:
            print(f'[Current_dataset] Pass current dataset!')
            continue
        # ------------------------------------------------------------------

        dataset  = HowTo100M(args                   = args,
                            valid_current_video_ids = valid_current_video_ids,
                            tokenizer               = tokenizer,
                            processor               = transform,
                            video_dict              = total_video_dict,
                            frames_h5               = frames_h5,)

        dataloader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            # collate_fn  = dataset.collate_fn,
            drop_last   = False,
            pin_memory  = True,  # better when training on GPU.
            shuffle     = False) # Don't need to shuffle for captioning

        print('Start batch')
        text_ids_dict = {}
        # text_feats_dict = {}
        # text_atts_dict = {}
        # image_feats_dict = {}
        clip_sim_dict = {}
        step = 0
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=config.fp16):
                pbar = tqdm(dataloader)
                for batch in pbar:
                    pbar.set_description(f"[{args.part:2d}/{args.total:2d}] [{start}:{end}({len(valid_current_video_ids)}) in {len(total_video_ids)} ({100*end/len(total_video_ids):.2f}%)] [#clips: {len(dataset)}]")
                    video_ids, text_ids, frames, raw_texts, valid_flag = batch
                    if np.sum(np.array(valid_flag)) == 0:
                        continue
                    video_ids = np.array(video_ids)[np.array(valid_flag)]

                    # retrieval_utils.py extract_text_feats() L18
                    raw_texts  = list(np.array(raw_texts)[np.array(valid_flag)])
                    text_ids   = np.array(text_ids)[np.array(valid_flag)]
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

                    for idx, v_id in enumerate(video_ids):
                        if v_id not in clip_sim_dict.keys():
                            text_ids_dict[v_id]    = [np.array(text_ids[idx])]
                            # text_feats_dict[v_id]  = [text_feats[idx]]
                            # text_atts_dict[v_id]   = [text_atts[idx]]
                            # image_feats_dict[v_id] = [image_feats[idx]]
                            clip_sim_dict[v_id]    = [similarity[idx]]
                        else:
                            text_ids_dict[v_id].extend([text_ids[idx]])
                            # text_feats_dict[v_id].extend([text_feats[idx]])
                            # text_atts_dict[v_id].extend([text_atts[idx]])
                            # image_feats_dict[v_id].extend([image_feats[idx]])
                            clip_sim_dict[v_id].extend([similarity[idx]])

                    step += 1

                for v_id in clip_sim_dict.keys():
                    text_ids_dict[v_id]    = np.vstack(text_ids_dict[v_id])
                    # text_feats_dict[v_id]  = np.vstack(text_feats_dict[v_id])
                    # text_atts_dict[v_id]   = np.vstack(text_atts_dict[v_id])
                    # image_feats_dict[v_id] = np.vstack(image_feats_dict[v_id])
                    clip_sim_dict[v_id]    = np.vstack(clip_sim_dict[v_id])

                save_embeds_sims_chunk_UMT_single(args, config, model,
                                    text_ids_dict,
                                    # text_feats_dict,
                                    # text_atts_dict,
                                    # image_feats_dict,
                                    clip_sim_dict,
                                    h5py_f,)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print(colorful.bold_pink("Thank you and Good Job Computer.").styled_string)
