import nltk
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

tagged_sentences = nltk.corpus.treebank.tagged_sents()
print("품사 태깅이 된 문장 개수: ", len(tagged_sentences))

# 첫번째 샘플만 출력해보겠습니다.
# print(tagged_sentences[0])

# 훈련을 위한 전처리
sentences, pos_tags = [], []
for tagged_sentence in tagged_sentences: # 3,914개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어들은 sentence에 품사 태깅 정보들은 tag_info에 저장한다.
    sentences.append(list(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    pos_tags.append(list(tag_info)) # 각 샘플에서 품사 태깅 정보만 저장한다.

# 각 문장 샘플에 대해서 단어는 sentences에, 태깅 정보는 pos_tags에 저장하였습니다.
# 첫번째 문장 샘플을 출력해보겠습니다.
# print(sentences[0])
# print(pos_tags[0])

print('샘플의 최대 길이 : %d' % max(len(l) for l in sentences))
print('샘플의 평균 길이 : %f' % (sum(map(len, sentences))/len(sentences)))
plt.hist([len(s) for s in sentences], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

# 위의 그래프는 대부분의 샘플의 길이가 150 이내며 대부분 0~50의 길이를 가지는 것을 보여줍니다.
# 이제 케라스 토크나이저를 통해서 정수 인코딩을 진행합니다.
# 우선 케라스 토크나이저를 다음과 같이 함수로 구현합니다.


def tokenize(samples):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(samples)
  return tokenizer

# 문장 데이터에 대해서는 src_tokenizer를,
# 레이블에 해당되는 품사 태깅 정보에 대해서는 tar_tokenizer를 사용합니다.
src_tokenizer = tokenize(sentences)
tar_tokenizer = tokenize(pos_tags)

# 단어 집합과 품사 태깅 정보 집합의 크기를 확인해보겠습니다.
vocab_size = len(src_tokenizer.word_index) + 1
tag_size = len(tar_tokenizer.word_index) + 1
print('단어 집합의 크기 : {}'.format(vocab_size))
print('태깅 정보 집합의 크기 : {}'.format(tag_size))

# 정수 인코딩 수행
# 문장 데이터에 대해서 정수 인코딩이 수행된 결과는 X_train,
# 품사 태깅 데이터에 대해서 정수 인코딩이 수행된 결과는 y_train에 저장되었습니다.
X_train = src_tokenizer.texts_to_sequences(sentences)
y_train = tar_tokenizer.texts_to_sequences(pos_tags)

# 정수 인코딩이 되었는지 확인을 위해 임의로 2번 인덱스 샘플을 출력해보겠습니다.
# print(X_train[:2])
# print(y_train[:2])

# 앞서 본 그래프에 따르면, 대부분의 샘플은 길이가 150 이내입니다.
# X에 해당되는 데이터 X_train의 샘플들과
# y에 해당되는 데이터 y_train 샘플들의 모든 길이를 임의로 150정도로 맞추어 보겠습니다.
# 케라스의 pad_sequences()를 사용합니다.
max_len = 150
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
y_train = pad_sequences(y_train, padding='post', maxlen=max_len)

# 모든 샘플의 길이가 150이 되었습니다. 훈련 데이터와 테스트 데이터를 8:2의 비율로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.2, random_state=777)

# 각 데이터에 대한 크기(shape)를 확인해보겠습니다.
print('훈련 샘플 문장의 크기 : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))
print('테스트 샘플 문장의 크기 : {}'.format(X_test.shape))
print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))