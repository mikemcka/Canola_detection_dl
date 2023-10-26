#!/usr/bin/env python
# coding: utf-8

# Import librariies
from fastai.vision.all import *
import torch
from ipywidgets import IntProgress
import glob
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import cv2, os
from pathlib import Path
import time, pickle
import argparse

# Define the input arguments that can be provided by the user
parser = argparse.ArgumentParser(
    prog="UNET_TRIALS",
    description="Train kfolds of UNET using a diff fold as validation each time -provided by the user")

parser.add_argument("val_fold")
args = parser.parse_args()

# Prepare datasets
codes = np.loadtxt("/group/pawsey0149/mmckay/singularity/codes.txt", dtype='str')
path = Path('/group/pawsey0149/mmckay/singularity/training_set')

# CUSTOM FUNCTIONS
def label_func(fn):
    path_mask = Path("/group/pawsey0149/mmckay/singularity/training_set/images_labels")
    return path_mask/f"{fn.stem.replace('image', 'mask')}{fn.suffix}"

def get_im_dictionary(path):
	import pickle
	with open('/group/pawsey0149/mmckay/singularity/folds_dictionary.pkl', 'rb') as f:
		loaded_dict= pickle.load(f)
	return[Path(i) for j in range(5) for i in loaded_dict[j]]

# Get the predictions from the validation set
def export_predictions(imlist, outdir):
	for im in imlist:
		filename = os.path.basename(im).replace("image", "prediction")
		filename = filename.replace("png", "npy")
		impred = learn.predict(im)
		impred = impred[0].detach().numpy()
		if not os.path.exists(outdir):
			os.makedirs(outdir)
		out_path = outdir + "/" + filename
		np.save(out_path, impred)

if __name__ == "__main__":
	print(args.val_fold)
	start_time = time.time() 
	with open('/group/pawsey0149/mmckay/singularity/folds_dictionary.pkl', 'rb') as f:
		loaded_dict= pickle.load(f) 

	# Define the fold used as validation for the model training
	valid_idx = [Path(i) for i in loaded_dict[int(args.val_fold)]]
	a = get_im_dictionary(None)
	validx= [a.index(i) for i in valid_idx]

	#load data
	weedt = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
		get_items = get_im_dictionary,
		get_y = label_func,
		splitter=IndexSplitter(validx),
		batch_tfms= [*aug_transforms(do_flip=True, flip_vert=True,
		max_rotate=30.0, min_zoom=1.0,
		max_zoom=1.5, max_lighting=0.5,
		max_warp=0.2, p_affine=0.75,
		p_lighting=0.75, xtra_tfms=None,
		size = 500, mode='bilinear',
		pad_mode='reflection', align_corners=True, batch=False,
		min_scale=1.0)])

	# create the data loader
	dls= weedt.dataloaders(path, path=path, bs=5)
	
	print(f"Loaded the dataloader in {time.time() - start_time} seconds")

	learn = unet_learner(dls, 
		resnet34,
		n_out=3,
		self_attention=True,
		normalize=True,
		pretrained=True,
		loss_func=DiceLoss(),
		opt_func=Adam,
		wd=None,
		metrics=[DiceMulti, JaccardCoeff]).to_fp32()

	learn.fine_tune(200,
		base_lr=0.001,
		cbs=[CSVLogger(fname=f'/group/pawsey0149/mmckay/singularity/unet_results/unet18/unet18fold{args.val_fold}', append=False),
        EarlyStoppingCallback (monitor='dice_multi', min_delta=0.001, patience=5), 
		SaveModelCallback(comp=np.less, fname=f"bestunet18_fold{args.val_fold}", with_opt=True)])

	print(f" Trained the model in {time.time() - start_time} seconds")

	learn.save(f'/group/pawsey0149/mmckay/singularity/unet_results/unet34/unet34fold{args.val_fold}_modelweights', with_opt=True, pickle_protocol=2)
	
	learn.export(fname=f'/group/pawsey0149/mmckay/singularity/unet_results/unet34/unet34fold{args.val_fold}_modelweights_exported.pkl',
                pickle_module=pickle,
		pickle_protocol=2)

	imtest = glob.glob("/group/pawsey0149/mmckay/singularity/holdout_set/images/*")
	export_predictions(imtest, f"/group/pawsey0149/mmckay/singularity/unet_results/unet34/unet34_preds/unet34fold{args.val_fold}_predictions")
	
	print(f"Exported predictions in {time.time() - start_time} seconds")


