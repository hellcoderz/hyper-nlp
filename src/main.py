import pprint
import random
from collections import defaultdict
import cPickle as pkl

import operator

import time

from scipy import spatial


class HyperNLP:
    def __init__(self):
        self.version = "1.0.2"
        self.vocab = {}
        self.labels = set([])
        self.total_counter = defaultdict(float)
        self.predicted_cells = defaultdict(lambda: defaultdict(float))
        self.total_transitions = set([])
        self.config = {"total_bits": 1000, "active_bits_prcentage": 2}
        self.active_bits_num = int(self.config["total_bits"] * self.config["active_bits_prcentage"] * 1.0 / 100)
        self.predicted_label_vectors = {}

    def normalize(self, text):
        return text \
            .strip() \
            .lower() \
            .replace(":", "").replace("?", "").replace(".", "").replace("'", "") \
            .replace("-", " ")

    def get_rand_vector(self):
        """return random list of activated cells"""
        active_cells = []
        for _ in range(self.active_bits_num):
            r = random.randint(0, 999)
            active_cells.append(r)
        return active_cells

    def phrase2vectors(self, phrase, label=[]):
        words = phrase.split() + label
        vectors = []
        for word in words:
            if word in self.vocab:
                vectors.append(self.vocab[word])
            else:
                vectors.append([])
        return vectors

    # def build_vocab(self, train_samples):
    #     """build a vocab of word => active cell index list"""
    #     total_words = set([])
    #     for tup in train_samples:
    #         self.labels.add(tup[1].upper())
    #         words = tup[0].split()
    #         for word in words:
    #             total_words.add(word)
    #     print "Total unique words =", len(total_words)
    #     print "Total labels =", self.labels
    #     for word in total_words:
    #         self.vocab[word] = self.get_rand_vector()
    #     for label in self.labels:
    #         self.vocab[label] = self.get_rand_vector()
    #         predicted_vector = [0.0] * self.config["total_bits"]
    #         for idx_cell in self.vocab[label]:
    #             predicted_vector[idx_cell] = 1.0
    #         self.predicted_label_vectors[label] = predicted_vector

    def build_vocab(self, train_samples):
        """build a vocab of word => combination of all label vectors"""
        total_words = set([])
        temp = defaultdict(lambda: set([]))
        for tup in train_samples:
            label = tup[1].upper()
            if label not in self.labels:
                self.labels.add(label)
                self.vocab[label] = self.get_rand_vector()
            words = tup[0].split()
            for word in words:
                total_words.add(word)
                temp[word].add(tup[1].upper())
        print "Total unique words =", len(total_words)
        print "Total labels =", self.labels
        for word in total_words:
            self.vocab[word] = list(set([idx for l in temp[word] for idx in self.vocab[l]]))
        for label in self.labels:
            predicted_vector = [0.0] * self.config["total_bits"]
            for idx_cell in self.vocab[label]:
                predicted_vector[idx_cell] = 1.0
            self.predicted_label_vectors[label] = predicted_vector

    def to_prob(self):
        for idx_i, value in self.predicted_cells.iteritems():
            sorted_values = sorted(value.items(), key=operator.itemgetter(1), reverse=True)
            for idx_j, _count in sorted_values:
                total_count = self.total_counter[idx_i]
                self.predicted_cells[idx_i][idx_j] = _count / total_count
                self.total_transitions.add((idx_i, idx_j))

    def get_damp_factor(self, i, j):
        """calculate and return damp factor based on the distance of the words in the sample"""
        diff = j - i
        damp_factor = 1.0
        for k in range(diff - 1):
            damp_factor /= 2
        return damp_factor

    def calculate(self, vec_i, vec_j, damp_factor=1.0):
        """calculate the strength between each cell from vec_i to each cell in vec_j"""
        for idx_i in vec_i:
            for idx_j in vec_j:
                self.total_counter[idx_i] += damp_factor
                self.predicted_cells[idx_i][idx_j] += damp_factor

    def train(self, fname, sampling=10000):
        """train model using damping factor each turn, 50% drop every time startng with 1.0"""

        print "loading samples..."
        data = set([])
        f = open(fname, "rb")
        for line in f:
            parts = line.strip().split("=>")
            data.add((self.normalize(parts[1].strip()), parts[0].strip()))

        data = list(data)
        print "Total unique samples =", len(data)
        print "picking", sampling, "samples..."
        samples = []
        if sampling is not None:
            for i in range(sampling):
                samples.append(data[random.randint(0, len(data) - 1)])
        else:
            samples = data

        print "Building Vocab..."
        self.build_vocab(samples)

        print "Starting training..."
        for d in samples:
            # print d
            vectors = self.phrase2vectors(d[0], [d[1]])
            for i, vec_i in enumerate(vectors[:-1]):
                for j, vec_j in enumerate(vectors[i + 1:]):
                    self.calculate(vec_i, vec_j, self.get_damp_factor(i, j))

        print "self.predicted_cells =", len(self.predicted_cells)
        print "Converting to probs..."
        self.to_prob()

    def predict(self, query):
        query = self.normalize(query)
        vectors = self.phrase2vectors(query)
        activated_cells = []
        predicted_cells = defaultdict(float)
        cell_overlap_count = defaultdict(lambda: 1)
        for i, vec_i in enumerate(vectors):

            if len(predicted_cells) > 0:
                max_prob = max(predicted_cells.values())
                prob_threshold = max_prob * 0.2
                temp = defaultdict(float)
                for idx_cell, prob in predicted_cells.iteritems():
                    if max_prob - prob > prob_threshold:
                        temp[idx_cell] = prob
                predicted_cells = temp

            activated_cells = list(set(vec_i + predicted_cells.keys()))
            for idx_a in activated_cells:
                if idx_a in self.predicted_cells:
                    for idx_p, prob in self.predicted_cells[idx_a].iteritems():
                        predicted_cells[idx_p] += prob

        # for idx_cell, cnt in predicted_cells.iteritems():
        #     predicted_cells[idx_cell] = cnt * 1.0 / cell_overlap_count[idx_cell]

        # sorted_predicted_cells = sorted(predicted_cells.items(), key=operator.itemgetter(1), reverse=True)
        # predicted_cells = dict((idx_cell, prob) for idx_cell, prob in sorted_predicted_cells[:self.active_bits_num])

        # predicted_vector = [0.0]*self.config["total_bits"]
        # for idx_cell, prob in predicted_cells.iteritems():
        #     predicted_vector[idx_cell] = prob

        max_prob = max(predicted_cells.values())
        prob_threshold = max_prob * 0.2
        temp = defaultdict(float)
        for idx_cell, prob in predicted_cells.iteritems():
            if max_prob - prob > prob_threshold:
                temp[idx_cell] = prob
        predicted_cells = temp

        result = {}
        for label in self.labels:
            label_vector = self.vocab[label]
            set_in = set(predicted_cells.keys()).intersection(set(label_vector))

            # comman cells activated
            result[label] = len(set_in)

            # # using cosine similarity
            # result[label] = 1 - spatial.distance.cosine(predicted_vector, self.predicted_label_vectors[label])

            # temp = {}
            # for idx_cell in set_in:
            #     temp[idx_cell] = predicted_cells[idx_cell]
            # if len(temp) > 0:
            #     result[label] = (sum(temp.values()) / len(temp)) * 1000

        return sorted(result.items(), key=operator.itemgetter(1), reverse=True)

    def predict2(self, query):
        query = self.normalize(query)
        vectors = self.phrase2vectors(query)
        activated_cells = []
        predicted_cells = defaultdict(float)
        cell_overlap_count = defaultdict(lambda: 1)
        for i, vec_i in enumerate(vectors):
            for idx_a in vec_i:
                for idx_p, prob in self.predicted_cells[idx_a].iteritems():
                    predicted_cells[idx_p] += prob

        if len(predicted_cells) == 0:
            return {}

        max_prob = max(predicted_cells.values())
        prob_threshold = max_prob * 0.2
        temp = defaultdict(float)
        for idx_cell, prob in predicted_cells.iteritems():
            if max_prob - prob > prob_threshold:
                temp[idx_cell] = prob
        predicted_cells = temp

        predicted_vector = [0.0]*self.config["total_bits"]
        for idx_cell, prob in predicted_cells.iteritems():
            predicted_vector[idx_cell] = prob

        result = {}
        for label in self.labels:
            # using cosine similarity
            result[label] = 1 - spatial.distance.cosine(predicted_vector, self.predicted_label_vectors[label])

        return sorted(result.items(), key=operator.itemgetter(1), reverse=True)

    def save(self):
        print "saving model.."
        self.total_counter = dict(self.total_counter)
        self.predicted_cells = dict(self.predicted_cells)
        pkl.dump(self, open("model.pkl", "wb"))

    def load(self):
        print "Loading model..."
        model = pkl.load(open("model.pkl", "rb"))
        if self.version == model.version:
            self.vocab = model.vocab
            self.predicted_cells = model.predicted_cells
            self.labels = model.labels
            self.total_counter = model.total_counter
            self.total_transitions = model.total_transitions
            self.config = model.config
            self.active_bits_num = model.active_bits_num
            self.predicted_label_vectors = model.predicted_label_vectors
        else:
            print "Model version mis-match..."


def build_model_test():
    nlp = HyperNLP()
    nlp.train("../data/data.train", sampling=10000)
    nlp.save()
    print "total transition learned =", len(nlp.total_transitions)


def test_queries():
    nlp = HyperNLP()
    nlp.load()
    print "total transition learned =", len(nlp.total_transitions)
    examples = [
        "watch harry potter",
        "mets stats",
        "show me action movies",
        "how is yankees doing",
        "fantasy rushing leaders",
        "how many offensive rebounds do the indians have",
        "seahawks lineup tonight",

        "Who is pitching for the phillies tonight",
        "What's the starting lineup for yesterday's twins game",
        "Who's starting for the mets tonight",
        "Who started for the mets yesterday",
        "Phillies lineup tonight",

        "How did the phillies play yesterday",
        "How did the twins do yesterday",
        "Did the Red Sox win yesterday",
        "Did the Diamondbacks lose yesterday",

        "Who had the most at bats",
        "Who lead in home runs",
        "Strikeout leaders",
        "Who lead the league in Saves",
        "Who had the best home era",
        "Who is leading the mlb in pitches thrown",
        "Who hit the most",
        "Who hit the most home runs",
        "Who has the most games played in the mlb",

        "Kyle Lowry stats",
        "Buster Posey",
        "Ortiz stats",
        "Reimold mlb stats",
        "What are Chris davis stats",
        "What's Chris Davis's batting average",
        "What's Gerrit Cole's era",
        "How many strikeouts does Gerrit Cole have",
        "How many hits does Mike Trout have",
        "How many times has escobar been hit by a pitch",
        "What is the on base percentage of jose altuve",
        "Ryan howard doubles",
        "how many times has edwin struckout",
        "Escobar royals stats",
        "Joseph hits stats",
        "Phillies number thirty three",
        "Who is #33 on the cubs",
        "Who is number 11 on the cubs",
        "Phillies #30",
        "Red Sox 20",
        "draymond green Offensive Rebounds",
        "White Sox twenty four doubles",
        "what is C.J. Anderson's longest rush",
        "How many three pointers has Steph Curry made",

        "George Springer versus Todd Frazier",
        "Yunel Escobar vs Dustin Pedroia",
        "homeruns Mike Trout vs Todd Frazier",
        "Mike Trout hits vs Todd Frazier",
        "George Springer versus Yunel Escobar runs",
        "George Springer vs Yunel Escobar runs",

        "What are the phillies standings",
        "How good are the mets",
        "Twins standings",
        "How are the dodgers doing",
        "NLE standings",

        "phillies vs mets",
        "Minnesota Twins versus Arizona Diamondbacks",

        "Phillies rank",
        "Twins rankings",
        "Mets batting rank",
        "What is the dodgers pitching rank",
        "What is the new york pitching rank",

        "Yankees stats",
        "Mets team stats",
        "What are the dodgers stats",
        "Red sox",
        "How many games played do the Phillies have",

        "Texas mlb stats",
        "How many runs do the phillies have",
        "What is the mets whip",
        "What is the eagles first down percentage",
        "Giants Extra Point Percentage",
        "How many average offensive rebounds do the knicks have",
        "how many offensive rebounds do the bulls have",
        "big bang theory",
        "cnn",
        "show me all action movies on demand"
    ]

    start = time.time()
    for ex in examples:
        print ">", ex, "=>", nlp.predict2(ex)
    end = time.time()
    diff = end - start
    print "Total Time Taken =", diff * 1000
    print "Time Taken per Query =", diff * 1000 / len(examples)


if __name__ == "__main__":
    build_model_test()
    # test_queries()


# for tup, prob in model["counter"].iteritems():
#     print tup, "=>", prob
