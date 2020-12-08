import re


def get_tokens(input_file_name, stopwords_file_name):
    input_file_content = get_file_content(input_file_name)
    stopwords = get_stopwords_from_file(stopwords_file_name)

    sentences = split_text_in_sentences(input_file_content)
    sentences = split_sentences_in_words(sentences)
    sentences = validate_words_from_sentences(sentences)
    sentences = remove_stopwords_from_sentences(sentences, stopwords)
    sentences = remove_empty_words(sentences)
    sentences = remove_empty_sentences(sentences)

    return sentences


def generate_one_hot_vectors(tokens):
    token_dict = {}
    for sentence in tokens:
        for token in sentence:
            if token not in token_dict:
                token_dict[token] = len(token_dict)

    return [
        [
            [(1 if i == token_dict[token] else 0) for i in range(len(token_dict))]
            for token in sentence
        ]
        for sentence in tokens
    ]


def pad_one_hot_sentences(one_hot_vectors):
    max_sentence_word_count = max([len(sentence) for sentence in one_hot_vectors])
    token_dict_len = len(one_hot_vectors[0][0])

    return [
        [
            (sentence[i] if i < len(sentence) else [0 for j in range(token_dict_len)])
            for i in range(max_sentence_word_count)
        ]
        for sentence in one_hot_vectors
    ]


def get_file_content(input_file_name):
    with open(input_file_name, 'r') as f:
        return f.read()


def split_text_in_sentences(text):
    return re.split(r'[.!?;\"]', text)


def split_sentences_in_words(sentences_list):
    return [re.split(r'\s', sentence) for sentence in sentences_list]


def validate_words_from_sentences(sentences_list):
    return [
        [
            re.sub(r'[^a-zA-Z]', '', word).lower()
            for word in sentence
        ]
        for sentence in sentences_list
    ]


def get_stopwords_from_file(stopwords_file_name):
    stopwords_list = []

    with open(stopwords_file_name, 'r') as stopwords_file:
        lines = stopwords_file.readlines()
        for line in lines:
            stopwords_list.append(line.strip().lower())

    return stopwords_list


def remove_stopwords_from_sentences(sentences_list, stopwords_list):
    return [
        [
            word for word in sentence if word not in stopwords_list
        ]
        for sentence in sentences_list
    ]


def remove_empty_words(sentences_list):
    return [
        [word for word in sentence if len(word) > 0]
        for sentence in sentences_list
    ]


def remove_empty_sentences(sentences_list):
    return [sentence for sentence in sentences_list if len(sentence) > 0]
