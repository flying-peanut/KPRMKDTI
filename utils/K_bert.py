import torch
import numpy as np
import pandas as pd
from my_nn import EarlyStopping, set_random_seed, BERT_atom_embedding_generator
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
set_random_seed()
args = {}
args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
args['metric_name'] = 'roc_auc'
args['batch_size'] = 128
args['num_epochs'] = 200
args['d_model'] = 768
args['n_layers'] = 6
args['vocab_size'] = 47
args['maxlen'] = 201
args['d_k'] = 64
args['d_v'] = 64
args['d_ff'] = 768 * 4
args['n_heads'] = 12
args['global_labels_dim'] = 1
args['atom_labels_dim'] = 15
args['lr'] = 3e-5
args['pretrain_layer'] = 6
args['mode'] = 'higher'
args['task_name'] = 'HIA'
args['patience'] = 20
args['times'] = 10
args['pretrain_model_drug'] = 'pretrain_k_bert_epoch_7.pth'



drug_model = BERT_atom_embedding_generator(d_model=args['d_model'], n_layers=args['n_layers'], vocab_size=args['vocab_size'],
                                        maxlen=args['maxlen'], d_k=args['d_k'], d_v=args['d_v'], n_heads=args['n_heads'], d_ff=args['d_ff'],
                                        global_label_dim=args['global_labels_dim'], atom_label_dim=args['atom_labels_dim'], use_atom=False)
stopper = EarlyStopping(pretrained_model=args['pretrain_model_drug'],
                        pretrain_layer=args['pretrain_layer'],
                        mode=args['mode'])
drug_model.to(args['device'])
stopper.load_pretrained_model(drug_model)


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    # return ' '.join(tokens)
    return tokens


def construct_input_from_smiles(smiles, max_len=200):
    try:
        # built a pretrain data from smiles
        atom_list = []
        atom_token_list = ['c', 'C', 'O', 'N', 'n', '[C@H]', 'F', '[C@@H]', 'S', 'Cl', '[nH]', 's', 'o', '[C@]',
                           '[C@@]', '[O-]', '[N+]', 'Br', 'P', '[n+]', 'I', '[S+]',  '[N-]', '[Si]', 'B', '[Se]', '[other_atom]']
        all_token_list = ['[PAD]', '[GLO]', 'c', 'C', '(', ')', 'O', '1', '2', '=', 'N', '3', 'n', '4', '[C@H]', 'F', '[C@@H]', '-', 'S', '/', 'Cl', '[nH]', 's', 'o', '5', '#', '[C@]', '[C@@]', '\\', '[O-]', '[N+]', 'Br', '6', 'P', '[n+]', '7', 'I', '[S+]', '8', '[N-]', '[Si]', 'B', '9', '[2H]', '[Se]', '[other_atom]', '[other_token]']

        word2idx = {}
        for i, w in enumerate(all_token_list):
            word2idx[w] = i
        token_list = smi_tokenizer(smiles)
        padding_list = ['[PAD]' for x in range(max_len-len(token_list))]
        tokens = ['[GLO]'] + token_list + padding_list
        atom_mask_list = []

        index = 0
        tokens_idx = []
        for i, token in enumerate(tokens):
            if token in atom_token_list:
                atom_mask_list.append(1)
                index = index + 1
                tokens_idx.append(word2idx[token])
            else:
                if token in all_token_list:
                    tokens_idx.append(word2idx[token])
                    atom_mask_list.append(0)
                elif '[' in list(token):
                    tokens[i] = '[other_atom]'
                    atom_mask_list.append(1)
                    index = index + 1
                    tokens_idx.append(word2idx['[other_atom]'])
                else:
                    tokens[i] = '[other_token]'
                    tokens_idx.append(word2idx['[other_token]'])
                    atom_mask_list.append(0)


        tokens_idx = [word2idx[x] for x in tokens]
        if len(tokens_idx) == max_len + 1:
            return tokens_idx, atom_mask_list
        else:
            return 0, 0
    except:
        return 0, 0


def extract_middle_smiles(smiles, length=200):
    if len(smiles) > length:
        start_idx = (len(smiles) - length) // 2  
        return smiles[start_idx:start_idx + length]
    return smiles

def check(smiles):
    ls = smi_tokenizer(smiles)
    if(len(ls) > 200):
        return False
    else:
        return True
    

def cut_smiles(smiles):
    if(check(smiles) == False):
        for i in range(200, 0, -1):
            new_smiles = extract_middle_smiles(smiles)
            if(check(new_smiles) == True):
                return new_smiles
        return ""
    return smiles

def bert_atom_embedding(smiles):
    token_idx, atom_mask_list = construct_input_from_smiles(smiles)
    token_idx = torch.tensor([token_idx]).long().to(args['device'])
    atom_mask = atom_mask_list
    atom_mask_np = np.array(atom_mask)
    atom_mask_index = np.where(atom_mask_np == 1)
    h_global, h_atom = drug_model(token_idx, atom_mask_index)
    h_global = h_global.cpu().squeeze().detach().numpy()
    h_atom = h_atom.cpu().squeeze().detach().numpy()
    return h_global, h_atom

def process_smiles(smiles):
    smiles = cut_smiles(smiles)
    return bert_atom_embedding(smiles)

def featurize_kbert(inpath, outpath):
    num = 0
    task_list = ['test', 'train']
    # task_list = ['test']
    for task_name in task_list:
        print(task_name)
        df = pd.read_csv(inpath + task_name + '.csv', index_col=None)
        # dataset = pd.read_csv('./dataset/sample/'+ task_name + '.csv', index_col=None)

        smiles_list = df['SMILES'].values.tolist()
        pretrain_features_list = []
        for i in range(len(smiles_list)):
            smiles = smiles_list[i]
            print("{}/{}".format(i+1, len(smiles_list)))
            try:
                h_smiles, g_atom = process_smiles(smiles)
                pretrain_features_list.append(h_smiles)
                    # print(h_global.shape)
            except Exception as e:
                print(e)
                num += 1
                pretrain_features_list.append(['NaN' for x in range(768)])
                # print(1)
        for i in range(len(pretrain_features_list[0])):
            global_feature_n = [pretrain_features_list[x][i] for x in range(len(pretrain_features_list))]
            df['k_bert_'+str(i+1)] = global_feature_n
        df = df[df['k_bert_1']!='NaN']
        df.to_csv(outpath + task_name + '.csv', index=False)
    print("problem:{}".format(num))

if __name__ == '__main__':
    inpath = './dataset/origin/'
    outpath = './dataset/new_featurized/k_bert/'
    featurize_kbert(inpath, outpath)


