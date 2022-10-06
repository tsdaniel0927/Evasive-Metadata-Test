from preprocess4wae import *
import re


def pad_zero(lst, max_len=-1):
    if max_len == -1:
        for i in lst:
            max_len = max(max_len, len(i))
    for cnt in range(len(lst)):
        lst[cnt] = lst[cnt] + [0] * (max_len - len(lst[cnt]))
        # 截断过长的部分；当max-len是手动设置的参数时，需要截断长度超过的部分
        lst[cnt] = lst[cnt][:max_len]
    return lst


def emoji_replace(text):
    emoji_replace_dict = {"😂": " emoji_a ", "😳": " emoji_b ", "👀": " emoji_c ",
                          "🙏": " emoji_d ", "❤": " emoji_e ", "👏": " emoji_f ",
                          "🙌": " emoji_g ", "😭": " emoji_h ", "😒": " emoji_i ",
                          "😩": " emoji_j ", "😷": " emoji_k ", "👍": " emoji_l ",
                          "😍": " emoji_m ", "🎉": " emoji_n ", "😫": " emoji_o ",
                          "😔": " emoji_p ", "💔": " emoji_q ", "😊": " emoji_r ",
                          "😁": " emoji_s ", "🙅": " emoji_t "}
    for k in emoji_replace_dict.keys():
        if k in text:
            text = text.replace(k, emoji_replace_dict[k])
    return text


def delete_special_freq(text):
    if len(text) < 20: return text
    # print('转换前：', text)
    raw = text
    text = [word for word in text.split() if word[0] != '@']
    text = ' '.join(text)
    # 如果只有@，则保留@
    if text == '': text = raw
    text = text.replace('URL', '')
    # 如果只有url，则保留url
    if text == '': text = raw
    # print('转换后：', text)
    # print()
    return text


def replace_url_with_token(url):
    # print("替换前：", url)
    url = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 'URL', url, flags=re.MULTILINE)
    # print("替换后：", url)
    return url


# 读取全部回复文本（1000000是上限；如果文件中超过上限，则后面的不处理，用于处理数据过多的文件，100000手动设置）
raw_tokenized_all_examples = load_data(100000, 'data/pathPheme5_texts.txt', True, "spacy")
examples_num = len(raw_tokenized_all_examples)
# 按照比例划分作为训练和验证集
# 复制一遍；便于打乱顺序等操作
random_all_examples = [line for line in raw_tokenized_all_examples]
random.shuffle(random_all_examples)
print("fitting count vectorizer...")

count_vectorizer = CountVectorizer(stop_words='english',
                                   max_features=5000,
                                   token_pattern=r'\b[^\d\W]{2,20}\b'
                                   )
# tfidf_vectorizer = TfidfVectorizer(stop_words='english',
#                                    max_features=args.vocab_size,
#                                    token_pattern=r'\b[^\d\W]{3,30}\b')

count_vectorizer.fit(tqdm(random_all_examples))

nlp = spacy.load('en_core_web_sm')
tokenizer = Tokenizer(nlp.vocab)

response_id2paths_dict = {}
random_id2paths_dict = {}
data_path = '220913__new_data/gen_test.json'
with tqdm(open(data_path, "r"), desc=f"loading {data_path}") as f:
    for line in f:
        example = json.loads(line.strip())
        tokenized_examples = []
        id_ = example['id_']
        response_tweets = example['tweets'][1:]
        for i in response_tweets:
            i = replace_url_with_token(i)
            i = emoji_replace(i)
            i = delete_special_freq(i)
            tokens = list(map(str, tokenizer(i)))
            text = ' '.join(tokens)
            tokenized_examples.append(text)

        vectorized_raw_all_examples = count_vectorizer.transform(tqdm(tokenized_examples))

        vectorized_raw_all_examples = sparse.hstack(
            (np.array([0] * len(tokenized_examples))[:, None], vectorized_raw_all_examples)).tocsc()
        response_id2paths_dict[id_] = [
            vectorized_raw_all_examples[i].nonzero()[1].tolist() for path_cnt, i in
            enumerate(range(len(response_tweets)))]
        # 将句子表示编码成相同的长度
        response_id2paths_dict[id_] = pad_zero(response_id2paths_dict[id_])

        random_tweets = example['tweets']
        for i in random_tweets:
            i = replace_url_with_token(i)
            i = emoji_replace(i)
            i = delete_special_freq(i)
            tokens = list(map(str, tokenizer(i)))
            text = ' '.join(tokens)
            tokenized_examples.append(text)

        vectorized_raw_all_examples = count_vectorizer.transform(tqdm(tokenized_examples))

        vectorized_raw_all_examples = sparse.hstack(
            (np.array([0] * len(tokenized_examples))[:, None], vectorized_raw_all_examples)).tocsc()
        random_id2paths_dict[id_] = [
            vectorized_raw_all_examples[i].nonzero()[1].tolist() for path_cnt, i in
            enumerate(range(len(random_tweets)))]
        # 将句子表示编码成相同的长度
        random_id2paths_dict[id_] = pad_zero(random_id2paths_dict[id_])

with open('220913__new_data/gen_test_dict_random.json', 'w') as f:
    json.dump(random_id2paths_dict, f)

with open('220913__new_data/gen_test_dict_response.json', 'w') as f:
    json.dump(response_id2paths_dict, f)
