import string
from nltk.tokenize import sent_tokenize as nlkt_sent_tokenize
from nltk.tokenize import word_tokenize as nlkt_word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.corpus import stopwords
import numpy as np


def cosine_distance(v1, v2):
    return 1 - (v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Calculates cosine similarity
def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine_distance(v1, v2)) + 1) / 2
    return score

def sent_tokenize(text):
    sents = nlkt_sent_tokenize(text)
    sents_filtered = []
    for s in sents:
        sents_filtered.append(s)
    return sents_filtered


def cleanup_sentences(text):
    stop_words = set(stopwords.words("english"))
    sentences = sent_tokenize(text)
    sentences_cleaned = []
    for sent in sentences:
        words = nlkt_word_tokenize(sent)
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if not w.lower() in stop_words]
        words = [w.lower() for w in words]
        sentences_cleaned.append(" ".join(words))
    return sentences_cleaned


def get_tf_idf(sentences):
    vectorizer = CountVectorizer()
    sent_word_matrix = vectorizer.fit_transform(sentences)

    transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
    tfidf = transformer.fit_transform(sent_word_matrix)
    tfidf = tfidf.toarray()

    centroid_vector = tfidf.sum(0)
    centroid_vector = np.divide(centroid_vector, centroid_vector.max())

    feature_names = vectorizer.get_feature_names_out()

    relevant_vector_indices = np.where(centroid_vector > 0.4)[0]

    word_list = list(np.array(feature_names)[relevant_vector_indices])
    return word_list

def build_embedding_representation(words, model):
    s = " ".join(words)
    embedding_representation = model.embedding_representation({"query": s})
    return embedding_representation

def summarize(text, embedding_model):
    raw_sentences = sent_tokenize(text)
    clean_sentences = cleanup_sentences(text)
    centroid_words = get_tf_idf(clean_sentences)
    centroid_vector = build_embedding_representation(centroid_words, embedding_model)
    sentences_scores = []
    for i in range(len(clean_sentences)):
        words = clean_sentences[i].split()
        sentence_vector = build_embedding_representation(words, embedding_model)
        score = similarity(sentence_vector, centroid_vector)
        sentences_scores.append((i, raw_sentences[i], score, sentence_vector))
    sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
    count = 0
    sentences_summary = []
    for s in sentence_scores_sort:
        if count > 40:
            break
        include_flag = True
        for ps in sentences_summary:
            sim = similarity(s[3], ps[3])
            if sim > 0.95:
                include_flag = False
        if include_flag:
            sentences_summary.append(s)
            count += len(s[1].split())

        sentences_summary = sorted(
            sentences_summary, key=lambda el: el[0], reverse=False
        )

    summary = "\n".join([s[1] for s in sentences_summary])
    return summary.strip()
