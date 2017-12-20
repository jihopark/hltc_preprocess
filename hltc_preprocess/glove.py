import numpy as np
import pickle

def create_vocab_txt(glove_path):
    with open("glove.vocab", "w") as w:
        with open(glove_path, "r") as f:
            for line in f:
                word = line.split()[0]
                w.write(word + "\n")
                print(word)
    print("DONE")

def vocab_to_matrix(vocab_path, glove_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        vocab_ = vocab["word2index"]

    embeddings = np.zeros((len(vocab_),200))
    glove = {}
    with open(glove_path, "r") as f:
        for line in f:
            splits = line.split()
            glove[splits[0]] = np.array(splits[1:])
        print("loaded glove")
    for word in vocab_.keys():
        try:
            embeddings[vocab_[word], :] = glove[word]
        except KeyError:
            print("key error " + word)
    np.save("./vocab_glove.npy", embeddings)

def load_vocab():
    return [line.rstrip() for line in open("glove.vocab", "r")]

