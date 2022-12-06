from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "James is working at Disney in London"
# 토큰화 후 품사 태깅
tokenized_sentence = pos_tag(word_tokenize(sentence))
print(tokenized_sentence)

'''
[('James', 'NNP'), ('is', 'VBZ'), ('working', 'VBG'), 
('at', 'IN'), ('Disney', 'NNP'), ('in', 'IN'), ('London', 'NNP')]
'''

# 개체명 인식
ner_sentence = ne_chunk(tokenized_sentence)
print(ner_sentence)

'''
(S
  (PERSON James/NNP)
  is/VBZ
  working/VBG
  at/IN
  (ORGANIZATION Disney/NNP)
  in/IN
  (GPE London/NNP))
'''

import re
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 전처리 수행
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/ukairia777/tensorflow-nlp-tutorial/main/12.%20Sequence%20Labeling/dataset/train.txt",
    filename="train.txt")

f = open('train.txt', 'r')
tagged_sentences = []
sentence = []

for line in f:
    if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
        if len(sentence) > 0:
            tagged_sentences.append(sentence)
            sentence = []
        continue
    splits = line.split(' ')  # 공백을 기준으로 속성을 구분한다.
    splits[-1] = re.sub(r'\n', '', splits[-1])  # 줄바꿈 표시 \n을 제거한다.
    word = splits[0].lower()  # 단어들은 소문자로 바꿔서 저장한다.
    sentence.append([word, splits[-1]])  # 단어와 개체명 태깅만 기록한다.

# 전체 샘플 개수를 확인해보겠습니다.
print("전체 샘플 개수: ", len(tagged_sentences))

# 훈련을 위한 훈련 데이터에서 단어에 해당되는 부분과 개체명 태깅 정보에 해당되는 부분을 분리
sentences, ner_tags = [], []
for tagged_sentence in tagged_sentences:  # 14,041개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence)  # 각 샘플에서 단어들은 sentence에 개체명 태깅 정보들은 tag_info에 저장.
    sentences.append(list(sentence))  # 각 샘플에서 단어 정보만 저장한다.
    ner_tags.append(list(tag_info))  # 각 샘플에서 개체명 태깅 정보만 저장한다.

# 각 문장 샘플에 대해서 단어는 sentences에 태깅 정보는 ner_tags에 저장
print('첫번째 샘플의 문장 :', sentences[0])
print('첫번째 샘플의 레이블 :', ner_tags[0])

# 전체 데이터의 길이 분포 확인
print('샘플의 최대 길이 : %d' % max(len(sentence) for sentence in sentences))
print('샘플의 평균 길이 : %f' % (sum(map(len, sentences)) / len(sentences)))
plt.hist([len(sentence) for sentence in sentences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# 샘플들의 길이가 대체적으로 0~40의 길이를 가지며,
# 특히 0~20의 길이를 가진 샘플이 상당한 비율을 차지하는 것을 보여줍니다.
# 길이가 가장 긴 샘플의 길이는 113입니다.

# 케라스 토크나이저를 통해서 정수 인코딩을 진행합니다.
# 이번에는 문장 데이터에 있는 모든 단어를 사용하지 않고 높은 빈도수를 가진 상위 약 4,000개의 단어만을 사용합니다.
vocab_size = 4000
src_tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
src_tokenizer.fit_on_texts(sentences)

tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(ner_tags)

# 문장 데이터에 대해서는 src_tokenizer를, 레이블에 해당되는 개체명 태깅 정보에 대해서는 tar_tokenizer를 사용합니다.
tag_size = len(tar_tokenizer.word_index) + 1
print('단어 집합의 크기 : {}'.format(vocab_size))
print('개체명 태깅 정보 집합의 크기 : {}'.format(tag_size))

# 정수 인코딩을 수행합니다.
# 문장 데이터에 대해서 정수 인코딩이 수행된 결과는 X_train,
# 개체명 태깅 데이터에 대해서 정수 인코딩이 수행된 결과는 y_train에 저장되었습니다.
X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(ner_tags)

# 정수 인코딩이 되었는지 확인을 위해 임의로 첫번째 샘플을 출력해보겠습니다.
print('첫번째 샘플의 문장 :', X_train[0])
print('첫번째 샘플의 레이블 :', y_train[0])

# 현재 문장 데이터에 대해서는 일부 단어가 'OOV'로 대체된 상황입니다.
# 이를 확인하기 위해 디코딩 작업을 진행해봅시다.
# 이를 위해 정수로부터 단어로 변환하는 index_to_word를 만듭니다.
index_to_word = src_tokenizer.index_word
index_to_ner = tar_tokenizer.index_word

# 정수 인코딩 된 첫번째 문장을 다시 디코딩해보겠습니다.
decoded = []
for index in X_train[0]:  # 첫번째 샘플 안의 각 정수로 변환된 단어에 대해서
    decoded.append(index_to_word[index])  # 단어로 변환

print('기존 문장 : {}'.format(sentences[0]))
print('빈도수가 낮은 단어가 OOV 처리된 문장 : {}'.format(decoded))

# 일부 단어가 'OOV'로 대체되었습니다.
# 앞서 본 그래프에 따르면, 대부분의 샘플은 길이가 70 이내입니다.
# X에 해당되는 데이터 X_train의 샘플들과
# y에 해당되는 데이터 y_train 샘플들의 모든 길이를 임의로 70정도로 맞추어 보겠습니다.
# 패딩을 진행합니다.
max_len = 70
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
y_train = pad_sequences(y_train, padding='post', maxlen=max_len)

# 훈련 데이터와 테스트 데이터를 8:2의 비율로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)

# 레이블에 해당하는 태깅 정보에 대해서 원-핫 인코딩을 수행합니다.
y_train = to_categorical(y_train, num_classes=tag_size)
y_test = to_categorical(y_test, num_classes=tag_size)

# 각 데이터에 대한 크기(shape)를 확인해보겠습니다.
print('훈련 샘플 문장의 크기 : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))
print('테스트 샘플 문장의 크기 : {}'.format(X_test.shape))
print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))

# 모델 구축
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.optimizers import Adam

embedding_dim = 128
hidden_units = 128

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_units, return_sequences=True)))
model.add(TimeDistributed(Dense(tag_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, epochs=8, validation_data=(X_test, y_test))

# 학습이 종료되었다면 테스트 데이터에 대한 정확도를 측정합니다.
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

# 실제로 맞추고 있는지를 임의의 테스트 샘플로부터(인덱스 10번) 직접 실제값과 비교해보겠습니다.
# index_to_word와 index_to_ner를 사용하여 테스트 데이터에 대한 예측값과 실제값을 비교 출력합니다.

i = 10 # 확인하고 싶은 테스트용 샘플의 인덱스.

# 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = model.predict(np.array([X_test[i]]))

# 확률 벡터를 정수 레이블로 변경.
y_predicted = np.argmax(y_predicted, axis=-1)

# 원-핫 벡터를 정수 인코딩으로 변경.
labels = np.argmax(y_test[i], -1)

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for word, tag, pred in zip(X_test[i], labels, y_predicted[0]):
    if word != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[word], index_to_ner[tag].upper(), index_to_ner[pred].upper()))