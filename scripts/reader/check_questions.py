#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to make and save model predictions on an input dataset."""

import os
import time
import torch
import argparse
import logging
import json
import numpy as np
from tqdm import tqdm
from drqa.reader import Predictor
from sklearn.manifold import TSNE

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, default=None,
                    help='SQuAD-like dataset to evaluate on')
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=('Expand dictionary to use all pretrained '
                          'embeddings in this file.'))
parser.add_argument('--out-dir', type=str, default='/tmp',
                    help=('Directory to write prediction file to '
                          '(<dataset>-<model>.preds)'))
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Example batching size')
parser.add_argument('--top-n', type=int, default=1,
                    help='Store top N predicted spans per example')
parser.add_argument('--official', action='store_true',
                    help='Only store single top span instead of top N list')
parser.add_argument('--n', type=int, default=100,
                    help='random sample of N questions')

args = parser.parse_args()
t0 = time.time()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(
    args.model,
    args.tokenizer,
    args.embedding_file,
    args.num_workers,
)
if args.cuda:
    predictor.cuda()

# ------------------------------------------------------------------------------
# Read in dataset and generate question embedding
# ------------------------------------------------------------------------------


num_points = 400
questions = []
with open(args.dataset) as f:
    data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                questions.append(qa['question'])

results = {}
sampled_questions = np.random.choice(questions,args.n)
embeddings = predictor.embed_questions(sampled_questions)
#from tensorflow examples/udacity/5_word2vec.ipynb

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(embeddings[1:num_points+1, :])

