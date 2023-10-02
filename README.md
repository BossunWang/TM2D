# TM2D: Bimodality Driven 3D Dance Generation via Music-Text Integration
## [[Project Page]](https://garfield-kh.github.io/TM2D/) [[Paper]](https://arxiv.org/abs/2304.02419)

## Python Virtual Environment

Anaconda is recommended to create this virtual environment.

```sh
conda create -f environment.yaml
conda activate tm2d
```

If you cannot successfully create the environment, here is a requirements.txt:
```sh
pip install -r requirements.txt
```

## Download Data & Pre-trained Models

### Datasets
Create a dataset folder to place pre-traine models:
```sh
mkdir ./dataset
```
download the prepared dataset from [[google drive]](https://drive.google.com/drive/folders/1gtIcMORHwEIM61bZi-gGbEZWVahrGzMq?usp=sharing)

### Pre-trained Models
Create a checkpoint folder to place pre-traine models:
```sh
mkdir ./checkpoints
```
download the checkpoint dataset from [[google drive]](https://drive.google.com/drive/folders/1gtIcMORHwEIM61bZi-gGbEZWVahrGzMq?usp=sharing)

Once finished, the file directory should look like this:  
  ```
  ./Bailando/                     # this folder contain only the mp3, please follow the original git for dataprocess.
  ./HumanML3D_24joint_60fps/      # this folder is for motion data preparation 
  ./tm2d_60fps/checkpoints/       # download from google drive
  ./tm2d_60fps/eval4bailando/     # download from google drive
  ./tm2d_60fps/dataset/
  ./tm2d_60fps/dataset/aistppml3d  ----> softlink to ./HumanML3D_24joint_60fps/aistppml3d
  ```

## Training Models (music2dance)

### Training motion discretizer 
#### HumanML3D and AIST++
```sh
python train_vq_tokenizer_v3.py --gpu_id 1 --name VQVAEV3_aistppml3d_motion_1003_d3 --dataset_name aistppml3d --batch_size 512 --n_resblk 3 --n_down=3 --start_dis_epoch 300 --window_size 128 --save_every_e 2
```

### Tokenizing all motion data for the following training
#### HumanML3D and AIST++ 
```sh
python tokenize_script_dance.py --gpu_id 0 --name VQVAEV3_aistppml3d_motion_1003_d3 --dataset_name aistppml3d --which_vqvae E0310 --n_resblk 3 --n_down=3 --window_size 128
```

### Training music2dance model:
#### AIST++
```sh
python train_a2d_transformer_v5.py --gpu_id 0 --name A2Dv5_1108n4_lr1e-4 --dataset_name aistppml3d --which_vqvae E0310 --tokenizer_name VQVAEV3_aistppml3d_motion_1003_d3 --tokenizer_name_motion VQVAEV3_aistppml3d_motion_1003_d3_E0310 --proj_share_weight --n_enc_layers 6 --n_dec_layers 6 --d_model 512 --d_inner_hid 2048 --lambda_a2d 1 --save_every_e 3 --max_epoch 50 --lr 1e-4
```

### Testing music2dance model:
#### test visualization 
```sh
python evaluate_atm_transformer_byatmv2_a2m_wild.py --name A2Dv5_1108n4_lr1e-4 --eval_mode vis --which_vqvae E0310 --which_epoch E0021 --slidwindow_overlap 1 --ext atmv2-a2d-wild-E0021-overlap1 --tokenizer_name VQVAEV3_aistppml3d_motion_1003_d3 --dataset_name aistppml3d --num_results 1 --repeat_times 1 --sample --text_file ./input.txt --t2m_v2 --proj_share_weight --n_enc_layers 6 --n_dec_layers 6 --n_down=3 --d_model 512 --d_inner_hid 2048  --tokenizer_name_motion VQVAEV3_aistppml3d_motion_1003_d3_E0310
```
#### test performance 
```sh
python eval4bailando/evaluate_music2dance_aistpp.py --gpu_id 0 --name A2Dv5_1108n4_lr1e-4 --eval_mode metric --which_vqvae E0310 --dataset_name aistppml3d --ext atmv2-a2d --tokenizer_name VQVAEV3_aistppml3d_motion_1003_d3 --tokenizer_name_motion VQVAEV3_aistppml3d_motion_1003_d3_E0310 --proj_share_weight --n_enc_layers 6 --n_dec_layers 6 --n_down=3 --d_model 512 --d_inner_hid 2048 --num_results 50 --repeat_times 1 --sample --eval_epoch 36
```

 
## Training Models (tm2d)

### Training motion discretizer 
#### HumanML3D and AIST++
```sh
python train_vq_tokenizer_v3.py --gpu_id 1 --name VQVAEV3_aistppml3d_motion_1003_d3 --dataset_name aistppml3d --batch_size 512 --n_resblk 3 --n_down=3 --start_dis_epoch 300 --window_size 128 --save_every_e 2
```

### Tokenizing all motion data for the following training
#### HumanML3D and AIST++ 
```sh
python tokenize_script_dance.py --gpu_id 0 --name VQVAEV3_aistppml3d_motion_1003_d3 --dataset_name aistppml3d --which_vqvae E0190 --n_resblk 3 --n_down=3 --window_size 128
```

### Training tm2d model:
#### HumanML3D and AIST++ 
```sh
python train_atm_transformer_v5.py --gpu_id 0 --name ATMv5_1028n2-REdcpu --dataset_name aistppml3d --which_vqvae E0190 --tokenizer_name VQVAEV3_aistppml3d_motion_1003_d3 --tokenizer_name_motion VQVAEV3_aistppml3d_motion_1003_d3_E0190 --proj_share_weight --n_enc_layers 4 --n_dec_layers 4 --d_model 512 --d_inner_hid 1024 --lambda_a2d 1e-1 --lambda_t2m 1 --save_every_e 5 --max_epoch 50 --lr 2e-4
```

### Testing music2dance model:
#### test visualization 
```sh
python evaluate_atm_transformer_byatmv5_lf3_withScoreV3.py --gpu_id 0 --name ATMv5_1028n2-REdcpu --eval_mode vis_all --which_vqvae E0190 --which_epoch E0020 --ext atmv5-lf3-1103a1-r1s-ws-E0020 --tokenizer_name VQVAEV3_aistppml3d_motion_1003_d3 --dataset_name aistppml3d --num_results 1 --repeat_times 1 --sample --text_file ./input.txt --proj_share_weight --n_enc_layers 4 --n_dec_layers 4  --n_down=3 --d_model 512 --d_inner_hid 1024 --tokenizer_name_motion VQVAEV3_aistppml3d_motion_1003_d3_E0190
```

 
