from gensim.models import Word2Vec
from sklearn.manifold import TSNE

import sys
import numpy as np

from preprocessing import get_tokens, generate_one_hot_vectors, pad_one_hot_sentences


if __name__ == '__main__':
    tokens = get_tokens('input_file.txt', 'stopwords.txt')

    one_hot_vectors = generate_one_hot_vectors(tokens)

    weighted_vectors = Word2Vec(tokens, min_count=1).wv

    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]

        print(arg + ':', end=' ')
        for similar_word in weighted_vectors.most_similar(positive=[arg], topn=10):
            print(similar_word[0], end=' ')

        print()

    one_hot_vectors = pad_one_hot_sentences(one_hot_vectors)
    one_hot_vectors = np.array(one_hot_vectors)
    one_hot_vectors = one_hot_vectors.reshape((
        len(one_hot_vectors),
        len(one_hot_vectors[0]) * len(one_hot_vectors[0][0])
    ))
    one_hot_vectors = TSNE(n_components=2).fit_transform(one_hot_vectors)
    print(one_hot_vectors)
