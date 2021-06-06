import fasttext


def read_txt_file(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            content = [_.strip() for _ in f.readlines()]
    qids, labels, texts = [], [], []
    
    for line in content:
        parts = line.split()
        qid, text, label = parts[0], ''.join(parts[1]),''.join(parts[2])
        qids.append(qid)
        labels.append(label)
        texts.append(text)

    return qids, labels, texts

def loadData(filename):
    file_path = filename
    qids, labels, texts = read_txt_file(file_path)
    return texts

def prepro_text(datas):
    sentences = []
    label = 1
    for index in range(80000):
        try:
            if(index<60000):
                label=0
            else:
                label=1
            sentences.append('__label__'+str(label) + " "+datas[index]) #将类别和文本拼接起来
        except Exception as index:
            print('text: %s is error' %(index))
            continue

    return sentences
    
# 将处理过的数据写入文件，编码格式是utf-8
def write_data(datas, file_name):
    print('writing data to fastText format...')
    with open(file_name, 'w', encoding='utf-8') as f:
        for senten in datas:
            # print(senten)
            f.write(senten+'\n')
    print('wirte done!')
    

if __name__=='__main__':
    
    texts=loadData("train_data.txt")
    
    data=prepro_text(texts)
    write_data(data,"new_train_data.txt")
    model = fasttext.train_supervised(input="new_train_data.txt", epoch=10, lr=0.1, wordNgrams=2, minCount=1, loss="softmax")
    
    model.save_model("model.bin")
    
    texts=loadData("validation_data_demo.txt")
    result=model.predict(texts)
    print(str(result[0][0])[-3])
    print('writing data to demo format...')
    with open("validation_data.txt", 'w', encoding='utf-8') as f:
        f.write("qids\ttexts\tlabel\n")
        for i in range(11):
            f.write("200000"+str(i)+"\t"+texts[i]+str("\t")+str(result[0][i])[-3]+str("\n"))
    print('wirte done!')
