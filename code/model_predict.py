import random
import kashgari
import os
from tensorflow.python import keras
# from kashgari.tasks.labeling import BiGRU_CRF_Model
# from kashgari.embeddings import BERTEmbedding
from kashgari.embeddings import BareEmbedding
#from kashgari.tasks.labeling import BiLSTM_CRF_Model
from kashgari.tasks.labeling import BiLSTM_Model
from kashgari.embeddings import WordEmbedding
import kashgari
from kashgari.callbacks import EvalCallBack

def load_dataset(train_file, max_sent_len=128):
    f = open(train_file, 'r', encoding='utf-8')
    train_lines = f.read().strip().split('\n')
    f.close()

    train_lines = [line.strip().split() for line in train_lines]

    # print(train_lines)

    train_pruned_lines = []

    # prun data set
    for train_line in train_lines:
        current_length = 0
        line = []
        for word in train_line:
            line.append(word)
            current_length += len(word)
            if current_length > max_sent_len:
                train_pruned_lines.append(line.copy())
                current_length = 0
                line = []
        if len(line) == 0:
            continue
        train_pruned_lines.append(line.copy())

    # train data
    train_sents, train_tags = [], []
    for i, line in enumerate(train_pruned_lines):
        cur_sent = ''.join(line)
        pos = 0
        cur_tag = [0] * (len(cur_sent))
        if len(cur_sent) == 0:
            # print(i,line)
            continue
        for word in line:
            if len(word) == 1:
                # single word
                cur_tag[pos] = 'S'
                pos += 1
            else:
                # more than one word
                cur_tag[pos] = 'B'
                cur_tag[pos + len(word) - 1] = "E"
                if len(word) > 2:
                    cur_tag[pos + 1:pos + len(word) - 1] = ['M'] * (len(word) - 2)
                pos = pos + len(word)

        train_sents.append(list(cur_sent))
        train_tags.append(cur_tag)

    return train_sents, train_tags

def main():
    train_path = '/home/qianlang/WordSeg-master/Data/train/data_generate_train.utf8'
    dev_path = '/home/qianlang/WordSeg-master/Data/train/data_generate_train.utf8'
    #test_path = '/home/qianlang/WordSeg-master/Data/test/data_generate_test.utf8'
    test_path = '/home/qianlang/地质数据.txt'
    test_path1 = '/home/qianlang/WordSeg-master/Data/gold/msr_test_gold.utf8'
    test_path2 = '/home/qianlang/WordSeg-master/Data/gold/pku_training_words.utf8'
    # dev_path = r'D:\Pycharm\Project\data_analyze\Data_processing\data_generate\data_generate_dev.utf8'

    #train_x, train_y = load_dataset(train_path)
    #dev_x, dev_y = load_dataset(dev_path)
    test_x, test_y = load_dataset(test_path2)
    path = r'/home/qianlang/d1-m2'

    txt_files = os.listdir(path)

# print(txt_files)

    text_x = []
    text_y = []

    for file in txt_files:
        sentence = ''
        tag = ''
        txt_path = os.path.join(path, file)
       #print(txt_path)
        f = open(txt_path, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
        # print(line.split())
            if len(line.split()) != 0:
                sentence += line.split()[0]
                tag += line.split()[1][0]
        text_x.append(list(sentence))
        text_y.append(list(tag.replace('I', 'M')))
    
    loaded_model = kashgari.utils.load_model("cws_wwm_bert_bigru_crf_2.h5")
    #y = loaded_model.predict(test_x)
    loaded_model.evaluate(test_x, test_y)
    #print(y)

if __name__ == '__main__':
    main()