import torch
import torch.nn as nn
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, HfArgumentParser)
from transformers.models.bert.configuration_bert import BertConfig
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field


def prediction(seq, tokenizer, model):
    x_feat = tokenizer(seq, return_tensors='pt')
    for k in x_feat:
        x_feat[k] = x_feat[k].cuda()
    with torch.no_grad():
        out = model(x_feat['input_ids'])
    prob = torch.softmax(out['logits'], dim=-1)
    return round(float(prob[0][1]), 5)

def test():
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained("zhihan1996/DNABERT-2-117M",
                                      trust_remote_code=True,
                                      config=config)
    model = model.cuda()
    
    #parser = HfArgumentParser(factor_arg)
    #tf_name = parser.parse_args_into_dataclasses()
    
    hts_seqs = dict()
    with open('./SMS_participants.fasta') as test:
        lines = test.readlines()
        for i in range(0, len(lines)):
            s = lines[i].strip()
            if s[0] == '>':
                key = s[1:]
            else:
                hts_seqs[key] = s

    hts_ids = list(hts_seqs.keys())
    hts_submit = dict()
    hts_submit['tags'] = hts_ids
    
    checks = dict()
    with open('./bests.txt') as best:
        lines = best.readlines()
        for i in range(len(lines)):
            s = lines[i].strip().split()
            checks[s[0]] = s[1]
                
    for tf in ['CAMTA1', 'LEUTX', 'MYF6', 'PRDM13']:
        #tf = tf_name[0].factor
        checkpoint = torch.load(f'./DNABERT_2/finetune/{tf}_out/checkpoint-{checks[tf]}/pytorch_model.bin',
                                weights_only=True)
        model.load_state_dict(checkpoint)
        model = model.eval().cuda()

        predictions = []
        for name, seq in tqdm(hts_seqs.items()):
            predictions.append(prediction(seq, tokenizer, model))
        #with open(f'./sms_predicts/SMS_{tf}.npy', 'wb') as f:
            #np.save(f, np.array(predictions))
        hts_submit[tf] = predictions

    hts_df = pd.DataFrame.from_dict(hts_submit).set_index('tags')
    hts_df.to_csv(f'./sms_predicts/1_BERT2_pbm.tsv', sep="\t")

if __name__ == "__main__":
    test()