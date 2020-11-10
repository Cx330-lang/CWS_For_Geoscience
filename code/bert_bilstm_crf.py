from tensorflow.python import keras
from kashgari.tasks.labeling import BiGRU_CRF_Model
from kashgari.embeddings import BERTEmbedding
import kashgari
from kashgari.callbacks import EvalCallBack

def load_dataset(train_file, max_sent_len=256):
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
    test_path = '/home/qianlang/WordSeg-master/Data/test/data_generate_test.utf8'
    # dev_path = r'D:\Pycharm\Project\data_analyze\Data_processing\data_generate\data_generate_dev.utf8'

    train_x, train_y = load_dataset(train_path)
    dev_x, dev_y = load_dataset(dev_path)
    test_x, test_y = load_dataset(test_path)

    bert_embed = BERTEmbedding('chinese_wwm_ext_L-12_H-768_A-12',task=kashgari.LABELING,sequence_length=128)

    model = BiGRU_CRF_Model(bert_embed)

    tf_board_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq=1000)

    # Build-in callback for print precision, recall and f1 at every epoch step
    eval_callback = EvalCallBack(kash_model=model,
                             valid_x=dev_x,
                             valid_y=dev_y,
                             step=5)


    model.fit(train_x, 
	train_y, 
	dev_x, 
	dev_y,
	batch_size=256,
	callbacks=[eval_callback, tf_board_callback])

    model.evaluate(test_x, test_y)

    model.save('cws.h5')


if __name__ == '__main__':
    main()