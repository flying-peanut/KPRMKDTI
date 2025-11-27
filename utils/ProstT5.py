from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
# Load the model
model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.float() if device.type=='cpu' else model.half()

def ProstT5_emb(sequence_list):

    # prepare your protein sequences/structures as a list.
    # Amino acid sequences are expected to be upper-case ("PRTEINO" below)
    # while 3Di-sequences need to be lower-case ("strctr" below).
    # replace all rare/ambiguous amino acids by X (3Di sequences do not have those) and introduce white-space between all sequences (AAs and 3Di)
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_list]

    # The direction of the translation is indicated by two special tokens:
    # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
    # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
    sequence_examples = [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s # this expects 3Di sequences to be already lower-case
                        for s in sequence_examples
                        ]

    # tokenize sequences and pad up to the longest sequence in the batch
    ids = tokenizer.batch_encode_plus(sequence_examples,
                                    add_special_tokens=True,
                                    padding="longest",
                                    return_tensors='pt').to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = model(
                ids.input_ids, 
                attention_mask=ids.attention_mask
                )
    embed = embedding_repr.last_hidden_state[0,1:] 
    embed_protein = embed.mean(dim=0).cpu().tolist()
    # print(embed_protein[0])

    return embed_protein


def featurize_ProstT5(inpath, outpath):
    num = 0
    task_list = ['test', 'train']
    # task_list = ['test']
    for task_name in task_list:
        print(task_name)
        dataset = pd.read_csv(inpath + task_name + '.csv', index_col=None)
        # dataset = pd.read_csv('./dataset/sample/' + task_name + '.csv', index_col= None)
        protein_list = dataset['Target Sequence'].values.tolist()
        pretrain_features_list = []
        for i in range(len(protein_list)):
            protein = protein_list[i]
            print("{}/{}".format(i+1, len(protein_list)))
            try:
                proteins = []
                proteins.append(protein)
                h_protein = ProstT5_emb(proteins)
                pretrain_features_list.append(h_protein)
            except Exception as e:
                print(e)
                num += 1
                # pretrain_features_list.append(['NaN' for x in range(2048)])
                pretrain_features_list.append(['NaN' for x in range(1024)])
                # print(1)
        for i in range(len(pretrain_features_list[0])):
            global_feature_n = [pretrain_features_list[x][i] for x in range(len(pretrain_features_list))]
            dataset['ProstT5_'+str(i + 1)] = global_feature_n
        dataset = dataset[dataset['ProstT5_1']!='NaN']

        dataset.to_csv(outpath + task_name + '.csv', index=False)
        # dataset.to_csv('./dataset/sample/' + task_name+'1.csv', index = False)
    print("problem:{}".format(num))

if __name__ == '__main__':
    inpath = './dataset/origin_data/'
    outpath = './dataset/prostT5/'
    featurize_ProstT5(inpath, outpath)
