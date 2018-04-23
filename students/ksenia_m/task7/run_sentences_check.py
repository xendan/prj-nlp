import nltk
import random
import pandas
from nltk.corpus import brown
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def prepare_data():
    def append_as_labeled(is_last, labeled_sent, sent):
        is_first_lower = random.random() > 0.25
        for i in range(len(sent) - 1):
            labeled_sent.append([sent[i].lower() if i == 0 and is_first_lower else sent[i], i == len(sent) - 2])
        if is_last:
            labeled_sent.append([sent[-1], False])

    def need_terminate(counter):
        return (counter == 2 and random.random() > 0.7) or (counter == 3 and random.random() > 0.3) or counter > 3

    #sentences = brown.sents(categories=['news', 'editorial', 'reviews', 'humor'])
    df = pandas.read_csv('../../../../sources/ted-talks/transcripts.csv')
    counter = 0
    labeled_sentences = []
    for index, row in df.iterrows():
        if index <=10:
            print(row['transcript'])
       for sent in sentences:
    #     counter += 1
    #     is_last = need_terminate(counter)
    #     append_as_labeled(is_last, labeled_sent, sent)
    #     if is_last:
    #         counter = 0
    #         labeled_sent = []
    #         labeled_sentences.append(labeled_sent)

    return train_test_split(labeled_sentences, test_size=0.2)


class NGrams:

    def __init__(self, use_lower=False):
        self.use_lower = use_lower
        self.trigrams = {}
        self.bigrams = {}
        self.unigrams = {}

    def calculate(self):

        def get_word(words, i):
            if i < 0:
                return '<S>'
            if i >= len(words):
                return '</S>'
            return words[i]

        def inc_key(key, data_dict):
            val = 0
            if key in data_dict:
                val = data_dict[key]
            data_dict[key] = val + 1

        sentences = brown.sents(categories=['adventure', 'fiction', 'hobbies', 'mystery'])

        for sent in sentences:
            for i in range(-2, len(sent) + 1):
                word1 = get_word(sent, i)
                word2 = get_word(sent, i + 1)
                word3 = get_word(sent, i + 2)
                inc_key(word1, self.unigrams)
                inc_key((word1, word2), self.bigrams)
                inc_key((word1, word2, word3), self.trigrams)

    # def get_log_prob_words_is_sentence(self, words):
    #
    #     def get_log_prob_last_word(words, last_state):
    #         log_sum = 0
    #         for i in range(len(words) - 2):
    #             log_sum += log_prob()
    #
    #     return max(get_log_prob_last_word(words, True), get_log_prob_last_word(words, False))
    #


def evaluate(model, data):
    y_true = []
    y_pred = []
    for sentence in data:
        annotated = model.annotate(sentence)
        for i in range(len(annotated)):
            y_true.append(sentence[i][1])
            y_pred.append(annotated[i][1])

    print(classification_report(y_true, y_pred, labels=['Is End', 'Not End']))


class AverageModel:

    def train(self, data):
        def get_avg_sent_len(sent):
            num = 0
            for i in range(len(sent)):
                if sent[i][1]:
                    num += 1
            return len(sent) / num

        total_len = 0
        for sent in data:
            total_len += get_avg_sent_len(sent)

        self.avg_len = int(total_len / len(data))

    def annotate(self, sentence):
        annotated = []
        print(self.avg_len)
        for i in range(len(sentence)):
            annotated.append([sentence[i][0], i != 0 and i % self.avg_len == 0])

        return annotated


class LogisticRegressionModel:

    def __init__(self):
        self.vec = DictVectorizer()
        self.logreg = LogisticRegression()

    def extract_features(self, i, sentence):

        def get_length():
            length = 0
            for j in range(i):
                length += len(sentence[i])
            return length

        def is_capital(word):
            if len(word) == 0:
                return False
            return word[0].isupper()

        features = dict()
        features["word"] = sentence[i][0].lower()
        # features["next_is_capitalized"] = is_capital(sentence[i + 1]) if i < len(sentence) - 1 else True
        # features["length_from_start"] = get_length()
        # features["i"] = i
        # features["nex"]
        return features

    def train(self, data):
        features, labels = [], []

        for sent in data:
            for i in range(len(sent)):
                features.append(self.extract_features(i, sent))
                labels.append(sent[i][1])

        # print("DATA:", features[0:20])
        # print(self.vec.fit_transform(features).toarray()[0:10])
        self.logreg.fit(self.vec.fit_transform(features).toarray(), labels)

    def annotate(self, sentence):
        annotated = []
        for i in range(len(sentence)):
            x = self.vec.transform(self.extract_features(i, sentence)).toarray()
            predicted = self.logreg.predict(x)
            annotated.append([sentence[i][0], predicted[0]])
        # print(annotated)
        return annotated

train, test = prepare_data()

print(test)

# log_reg_model = LogisticRegressionModel()
# log_reg_model.train(train)
# evaluate(log_reg_model, test)
#
# model_on_average = AverageModel()
# model_on_average.train(train)
#
# evaluate(model_on_average, test)
# ngrams = NGrams()
# ngrams.calculate()

