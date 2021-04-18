import os
import pandas

#==========================================Load txt files======================================#
path = r'./bbc/business/'
files = os.listdir(path)
tile_list = []
data = pandas.DataFrame()
cont_list = []
for file in files:
    position = path + file
    with open(position, mode = 'r', encoding='ISO-8859-1') as f:
        content = f.readlines()
        content.remove('\n')
        tile_list.append(content[0].strip('\n'))
        content1 = [i.strip('\n') for i in content[1:]]
        cont_list.append(content1)

data['title'] = pandas.Series(tile_list)
data['content'] = pandas.Series(cont_list)
data['label'] = 'business'
data.to_csv('./part2_corpus/business.csv', header=True, index=False, encoding='utf_8_sig')

path = r'./bbc/entertainment/'
files = os.listdir(path)
tile_list = []
data = pandas.DataFrame()
cont_list = []
for file in files:
    position = path + file
    with open(position, mode = 'r', encoding='ISO-8859-1') as f:
        content = f.readlines()
        content.remove('\n')
        tile_list.append(content[0].strip('\n'))
        content1 = [i.strip('\n') for i in content[1:]]
        cont_list.append(content1)

data['title'] = pandas.Series(tile_list)
data['content'] = pandas.Series(cont_list)
data['label'] = 'entertainment'
data.to_csv('./part2_corpus/entertainment.csv', header=True, index=False, encoding='utf_8_sig')

path = r'./bbc/politics/'
files = os.listdir(path)
tile_list = []
data = pandas.DataFrame()
cont_list = []
for file in files:
    position = path + file
    with open(position, mode = 'r', encoding='ISO-8859-1') as f:
        content = f.readlines()
        content.remove('\n')
        tile_list.append(content[0].strip('\n'))
        content1 = [i.strip('\n') for i in content[1:]]
        cont_list.append(content1)

data['title'] = pandas.Series(tile_list)
data['content'] = pandas.Series(cont_list)
data['label'] = 'politics'
data.to_csv('./part2_corpus/politics.csv', header=True, index=False, encoding='utf_8_sig')

path = r'./bbc/sport/'
files = os.listdir(path)
tile_list = []
data = pandas.DataFrame()
cont_list = []
for file in files:
    position = path + file
    with open(position, mode = 'r', encoding='ISO-8859-1') as f:
        content = f.readlines()
        content.remove('\n')
        tile_list.append(content[0].strip('\n'))
        content1 = [i.strip('\n') for i in content[1:]]
        cont_list.append(content1)

data['title'] = pandas.Series(tile_list)
data['content'] = pandas.Series(cont_list)
data['label'] = 'sport'
data.to_csv('./part2_corpus/sport.csv', header=True, index=False, encoding='utf_8_sig')

path = r'./bbc/tech/'
files = os.listdir(path)
tile_list = []
data = pandas.DataFrame()
cont_list = []
for file in files:
    position = path + file
    with open(position, mode = 'r', encoding='ISO-8859-1') as f:
        content = f.readlines()
        content.remove('\n')
        tile_list.append(content[0].strip('\n'))
        content1 = [i.strip('\n') for i in content[1:]]
        cont_list.append(content1)

data['title'] = pandas.Series(tile_list)
data['content'] = pandas.Series(cont_list)
data['label'] = 'tech'
data.to_csv('./part2_corpus/tech.csv', header=True, index=False, encoding='utf_8_sig')

#==========================================merge into bbc_news.csv======================================#
data1 = pandas.read_csv('./part2_corpus/business.csv')
print(data1.shape)
data2 = pandas.read_csv('./part2_corpus/entertainment.csv')
data3 = pandas.read_csv('./part2_corpus/politics.csv')
data4 = pandas.read_csv('./part2_corpus/sport.csv')
data5 = pandas.read_csv('./part2_corpus/tech.csv')

data1 = data1.append(data2, ignore_index=True)
print(data1.shape)
data1 = data1.append(data3, ignore_index=True)
print(data1.shape)
data1 = data1.append(data4, ignore_index=True)
print(data1.shape)
data1 = data1.append(data5, ignore_index=True)
print(data1.shape)
data1.to_csv('./part2_corpus/bbc_news.csv', header=True, index=False, encoding='utf_8_sig')

