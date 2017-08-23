import os
import time
import torch
import argparse
import logging
import json
import numpy as np
from tqdm import tqdm
from drqa.reader import Predictor
from drqa.reader.vector import vectorize, batchify, vectorize_question, batchify_questions
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from drqa.reader import layers



logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

torch.cuda.set_device(-1)

predictor = Predictor(None,None,None,None)
predictor.cuda()

num_points = 400
questions = []

with open('/data/drqa/data/datasets/SQuAD-v1.1-dev.json') as f:
    data = json.load(f)['data']
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                questions.append(qa['question'])

results = {}
sampled_questions = np.random.choice(questions,10)
#embeddings = predictor.embed_questions(sampled_questions)
tokenized = predictor.tokenize_questions(sampled_questions)
qdict = tokenized[0]
vq = vectorize_question(qdict,predictor.model)
bq = batchify_questions([vectorize_question(q, predictor.model) for q in tokenized])
embeddings = predictor.model.get_question_embeddings(bq)


