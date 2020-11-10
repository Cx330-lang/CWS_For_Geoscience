#coding=gbk
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')
# 这段代码的作用就是把标准输出的默认编码修改为gb18030，也就是与cmd显示编码GBK相同。

from nlpcda import Similarword
import pandas as pd
import numpy as np
import time
import random
from data_translate import BaiduTranslate
from fuzzywuzzy import fuzz

class DataAugment:

    def __init__(self, domain_path, generic_path, data_path, dict_path, translate):
        self.domain_path = domain_path
        self.generic_path = generic_path
        self.data_path = data_path
        self.baidu_trans = translate
        # self.dcit_path = dict_path
        self.df_dict_words = pd.read_table(dict_path)

        self.init_word_list()

    def init_word_list(self):
        self.domain_words = pd.read_table(r'./data_for_aug/new_words.txt')['words'].tolist()
        self.df = pd.read_excel(r'./data_for_aug/Chinese_English_word.xlsx')
        self.df.drop_duplicates()
        
        slef.data_f = open(self.data_path, 'a', encoding='utf-8')

    def get_universal_list(self):
        with open(self.generic_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        universal_word_lists = []
        for line in lines:
            universal_word_lists.extend(list(set(line.split())))

        return list(set(universal_word_lists))

    def get_domain_words(self):
        generic_list = self.get_universal_list()

        domain_list = []
        with open(self.domain_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line_list = line.strip().split()
            for word in line_list:
                if word in generic_list:
                    continue
                else:
                    domain_list.append(word)

        return list(set(domain_list))

    def data_generate(self):
        domain_words = self.get_domain_words()

        smw = Similarword(create_num=10, change_rate=0.8)
        smw.add_words(domain_words)

        with open(self.domain_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            rs1 = smw.replace(line)
            for s in rs1:
                self.data_f.write(s + '\n')
                
        self.data_f.close()

    def get_domain_word_dicts(self):
        # pass
        # domain_words = self.get_domain_words()

        # for word in self.domain_words:
        #     print(word)

        with open(self.domain_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        domain_word_dict = {}

        for line in lines:
            # print(line)
            line_list = line.split()
            for word in line_list:
                # print(word)
                if word in self.domain_words:
                    # print(word)
                    word_english = self.baidu_trans.BdTrans(word)
                    if word_english is not None:
                        for chinese_word, english_word in zip(self.df['word_chinese'], self.df['word_english']):
                            # print(chinese_word, english_word)
                            if english_word is not np.nan and not isinstance(english_word, int) and len(str(english_word)) >= 5:
                                smi1 = fuzz.partial_ratio(english_word, word_english)
                                if smi1 >= 85:
                                    if word not in domain_word_dict.keys():
                                        domain_word_dict[word] = set()
                                        domain_word_dict[word].add(chinese_word)
                                    else:
                                        domain_word_dict[word].add(chinese_word)

        return domain_word_dict

    def domain_data_generate(self):
    
        self.data_generate()
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), "../..")), "data_aug\领域增强数据.txt")

        domain_f_set = set()

        domain_word_dict = self.get_domain_word_dicts()
        with open(path, 'a+', encoding='utf-8') as domain_f:
            for line in lines:
                for word in line.split():
                    if word in domain_word_dict.keys():
                            sentence = line.replace(word, random.sample(domain_word_dict[word], 1)[0], 1)
                            if sentence not in domain_f_set:
                                domain_f_set.add(sentence)
                                domain_f.write(sentence)

def main():
    
    # 注意需要更改路径
    data_domain_path = r'/home/test/data_2/data_split_train.utf8'
    data_generic_path = r'/home/test/data_2/msr_training.utf8'
    data_augment_path = r'/home/test/data_2/增强数据.txt'
    domain_dict_path = r'/home/test/data_2/new_words.txt'

    Bt = BaiduTranslate('zh', 'en')

    da = DataAugment(data_domain_path, data_generic_path, data_augment_path, domain_dict_path, Bt)

    da.domain_data_generate()

if __name__ == '__main__':
    main()
