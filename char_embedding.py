# char_vocab 만들기
words = list(set(data["Word"].values))
chars = set([w_i for w in words for w_i in w])
chars = sorted(list(chars))
print('문자 집합 :',chars)

char_to_index = {c: i + 2 for i, c in enumerate(chars)}
char_to_index["OOV"] = 1
char_to_index["PAD"] = 0

index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key


max_len_char = 15

# 문자 시퀀스에 대한 패딩하는 함수
def padding_char_indice(char_indice, max_len_char):
  return pad_sequences(
        char_indice, maxlen=max_len_char, padding='post', value = 0)

# 각 단어를 문자 시퀀스로 변환 후 패딩 진행
def integer_coding(sentences):
  char_data = []
  for ts in sentences:
    word_indice = [word_to_index[t] for t in ts]
    char_indice = [[char_to_index[char] for char in t]
                                          for t in ts]
    char_indice = padding_char_indice(char_indice, max_len_char)

    for chars_of_token in char_indice:
      if len(chars_of_token) > max_len_char:
        continue
    char_data.append(char_indice)
  return char_data

# 문자 단위 정수 인코딩 결과
X_char_data = integer_coding(sentences)


# 정수 인코딩 이전의 기존 문장
print('기존 문장 :',sentences[0])

# 단어 단위 정수 인코딩 + 패딩
print('단어 단위 정수 인코딩 :')
print(X_data[0])

# 문자 단위 정수 인코딩
print('문자 단위 정수 인코딩 :')
print(X_char_data[0])

X_char_data = pad_sequences(X_char_data, maxlen=max_len, padding='post', value = 0)

X_char_train, X_char_test, _, _ = train_test_split(X_char_data, y_data, test_size=.2, random_state=777)

X_char_train = np.array(X_char_train)
X_char_test = np.array(X_char_test)

print(X_train[0])
print(index_to_word[150])
print(' '.join([index_to_char[index] for index in X_char_train[0][0]]))

print('훈련 샘플 문장의 크기 : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))
print('훈련 샘플 char 데이터의 크기 : {}'.format(X_char_train.shape))
print('테스트 샘플 문장의 크기 : {}'.format(X_test.shape))
print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))