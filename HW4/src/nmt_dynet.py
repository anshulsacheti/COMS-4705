from dynet import *
import argparse
from utils import Corpus
import random
import numpy as np
from bleu import get_bleu_score
import json
# import pdb

RNN_BUILDER = GRUBuilder

class nmt_dynet:

    def __init__(self, src_vocab_size, tgt_vocab_size, src_word2idx, src_idx2word, tgt_word2idx, tgt_idx2word, word_d, gru_d, gru_layers):

        # initialize variables
        self.gru_layers = gru_layers
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_word2idx = src_word2idx
        self.src_idx2word = src_idx2word
        self.tgt_word2idx = tgt_word2idx
        self.tgt_idx2word = tgt_idx2word
        self.word_d = word_d
        self.gru_d = gru_d

        self.model = Model()

        # the embedding parameters
        self.source_embeddings = self.model.add_lookup_parameters((self.src_vocab_size, self.word_d))
        self.target_embeddings = self.model.add_lookup_parameters((self.tgt_vocab_size, self.word_d))

        # YOUR IMPLEMENTATION GOES HERE
        # project the decoder output to a vector of tgt_vocab_size length
        self.output_w = self.model.add_parameters((tgt_vocab_size,gru_d))
        self.output_b = self.model.add_parameters((tgt_vocab_size))

        # encoder network
        # the foreword rnn
        self.fwd_RNN = RNN_BUILDER(gru_layers, word_d, gru_d/2, self.model)

        # the backward rnn
        self.bwd_RNN = RNN_BUILDER(gru_layers, word_d, gru_d/2, self.model)

        # decoder network
        self.dec_RNN = RNN_BUILDER(gru_layers, gru_d+word_d, gru_d, self.model)

        # raise NotImplementedError


    def encode(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return encoding of the source sentence
        '''

        embeddedWords = [self.source_embeddings[self.src_word2idx[word]] for word in src_sentence]
        # YOUR IMPLEMENTATION GOES HERE

        #Forward RNN Encoding
        f_states = []
        fwd_state = None
        fwd_is = self.fwd_RNN.initial_state()
        for i in embeddedWords:
            fwd_is = fwd_is.add_input(i)
            fwd_state = fwd_is.output()
            f_states.append(fwd_state)

        #Backward RNN Encoding
        b_states = []
        bwd_state = None
        bwd_is = self.bwd_RNN.initial_state()
        for i in reversed(embeddedWords):
            bwd_is = bwd_is.add_input(i)
            bwd_state = bwd_is.output()
            b_states.append(bwd_state)

        #b_states = list(reversed(b_states))

        #concat_states = [concatenate([f_states[i], b_states[len(f_states)-1-i]]) for i in range(len(f_states))]
        concat_states = [concatenate([f_states[i], b_states[i]]) for i in range(len(f_states))]
        return concat_states
        #   raise NotImplementedError

    def setDecoderState(self, decoder_rnn, encodeStates):
        '''
        decoder_rnn: decoder rnn object
        encodeStates: encoder hidden states
        return decoder_rnn initialized
        '''

        tmp = []
        last_encodeState = encodeStates[-1]
        for layer in range(self.gru_layers):
            tmp.append(last_encodeState)

        return decoder_rnn.initial_state(tmp)

    def get_loss(self, src_sentence, tgt_sentence):
        '''
        src_sentence: words in src sentence
        tgt_sentence: words in tgt sentence
        return loss for this source target sentence pair
        '''
        renew_cg()
        #Return encoding from encode
        encode_states = self.encode(src_sentence)
        last_encodeState = encode_states[-1]

        d = self.setDecoderState(self.dec_RNN,encode_states)
        # d = self.dec_RNN.initial_state()

        output_w = parameter(self.output_w)
        output_b = parameter(self.output_b)

        # dec_is = self.dec_RNN.initial_state()

        # YOUR IMPLEMENTATION GOES HERE

        #Word Embeddings
        tgt_embeddings = [self.target_embeddings[self.tgt_word2idx[word]] for word in tgt_sentence]

        #Update src/tgt sentences to use UNK for any new words
        #Dev set can have words not seen in data set that get embedded into <s>
        #Set them all the UNK
        # print("Pre - srcSentence: %s, tgtSentence: %s" % (" ".join(src_sentence), " ".join(tgt_sentence)))
        for i,word in enumerate(src_sentence):
            keys = self.src_word2idx.keys()
            if word not in keys:
                # print("Pre - srcSentence: %s" % (" ".join(src_sentence)))
                src_sentence[i]="<unk>"
                # print("Post - srcSentence: %s" % (" ".join(src_sentence)))

        for i,word in enumerate(tgt_sentence):
            keys = self.tgt_word2idx.keys()
            if word not in keys:
                tgt_sentence[i]="<unk>"
        # print("Post - srcSentence: %L, tgtSentece: %L" % (" ".join(src_sentence), " ".join(tgt_sentence)))

        #Concat y_i-1 word with bidir output as decoder input
        #Run through each decoder stage calculating loss against target word

        #Decoder states
        d_states = []
        loss = []

        tgt_sentence = [self.tgt_word2idx[e] for e in tgt_sentence]

        #calculate loss
        for word,nextWord in zip(tgt_sentence, tgt_sentence[1:]):

            #Simplify to dynet expression
            tgt_wordEmbed = concatenate([self.target_embeddings[word], last_encodeState])

            d = d.add_input(tgt_wordEmbed)

            #Neg log softmax
            p = softmax(output_w*d.output() + output_b)

            # score = pickneglogsoftmax(sum,self.tgt_word2idx[tgtWord])
            loss.append(-log(pick(p, nextWord)))
        loss = esum(loss)
            #tgt_wordEmbed = tgt_embeddings[i]


        # To compute the loss, you will use decoder’s previous state,
        # output of the encoder and the previous word to predict the current word.
        # The loss will then be computed based on the predicted and actual words
        # in the target sentence.

        return loss
        #raise NotImplementedError

    def generate(self, src_sentence):
        '''
        src_sentence: list of words in the source sentence (i.e output of .strip().split(' '))
        return list of words in the target sentence
        '''
        renew_cg()

        #Return encoding from encode
        encode_states = self.encode(src_sentence)
        last_encodeState = encode_states[-1]

        #Run input to encoder
        d = self.setDecoderState(self.dec_RNN,encode_states)
        # d = self.dec_RNN.initial_state()

        # YOUR IMPLEMENTATION GOES HERE

        #Update src/tgt sentences to use UNK for any new words
        # print("Pre - srcSentence: %L, tgtSentece: %L" % (" ".join(src_sentence), " ".join(tgt_sentence)))
        for i,word in enumerate(src_sentence):
            keys = self.src_word2idx.keys()
            if word not in keys:
                src_sentence[i]="<unk>"

        output_w = parameter(self.output_w)
        output_b = parameter(self.output_b)

        #Word Embeddings
        embedding = self.target_embeddings[self.tgt_word2idx["<s>"]]
        tgt_wordEmbed = concatenate([embedding, last_encodeState])
        d = d.add_input(tgt_wordEmbed)

        tgt_sentence = ["<s>"]
        i=1

        #keep adding new words to sentence until we add EOS. Can get into loops so max out at arbitrary size multiple
        while True and i<5*len(src_sentence):

            wordVec = softmax(output_w*d.output() + output_b).vec_value()
            newWordIndex = wordVec.index(max(wordVec))
            tgt_sentence.append(self.tgt_idx2word[newWordIndex])

            # End of sentence
            if self.tgt_idx2word[newWordIndex]=="</s>":
                break

            embedding = self.target_embeddings[newWordIndex]
            tgt_wordEmbed = concatenate([embedding, last_encodeState])
            d = d.add_input(tgt_wordEmbed)
            i+=1

        if i>=5*len(src_sentence):
            tgt_sentence.append("</s>")

        # raise NotImplementedError
        # print(tgt_sentence)
        return tgt_sentence

    def translate_all(self, src_sentences):
        translated_sentences = []
        for src_sentence in src_sentences:
            # print src_sentence
            translated_sentences.append(self.generate(src_sentence))

        return translated_sentences

    # save the model, and optionally the word embeddings
    def save(self, filename):

        self.model.save(filename)
        embs = {}
        if self.src_idx2word:
            src_embs = {}
            for i in range(self.src_vocab_size):
                src_embs[self.src_idx2word[i]] = self.source_embeddings[i].value()
            embs['src'] = src_embs

        if self.tgt_idx2word:
            tgt_embs = {}
            for i in range(self.tgt_vocab_size):
                tgt_embs[self.tgt_idx2word[i]] = self.target_embeddings[i].value()
            embs['tgt'] = tgt_embs

        if len(embs):
            with open(filename + '_embeddings.json', 'w') as f:
                json.dump(embs, f)

def get_val_set_loss(network, val_set):
        loss = []
        for src_sentence, tgt_sentence in zip(val_set.source_sentences, val_set.target_sentences):
            loss.append(network.get_loss(src_sentence, tgt_sentence).value())
        return sum(loss)

def main(train_src_file, train_tgt_file, dev_src_file, dev_tgt_file, model_file, num_epochs, embeddings_init = None, seed = 0):
    print('reading train corpus ...')
    train_set = Corpus(train_src_file, train_tgt_file)
    # assert()
    print('reading dev corpus ...')
    dev_set = Corpus(dev_src_file, dev_tgt_file)

    print('Initializing simple neural machine translator:')
    # src_vocab_size, tgt_vocab_size, tgt_idx2word, word_d, gru_d, gru_layers
    encoder_decoder = nmt_dynet(len(train_set.source_word2idx), len(train_set.target_word2idx), train_set.source_word2idx,
                                    train_set.source_idx2word, train_set.target_word2idx, train_set.target_idx2word, 50, 50, 2)

    trainer = SimpleSGDTrainer(encoder_decoder.model)

    sample_output = np.random.choice(len(dev_set.target_sentences), 5, False)
    losses = []
    best_bleu_score = 0
    for epoch in range(num_epochs):
        print('\n-------------------\nStarting epoch', epoch)
        # shuffle the training data
        combined = list(zip(train_set.source_sentences, train_set.target_sentences))
        random.shuffle(combined)
        train_set.source_sentences[:], train_set.target_sentences[:] = zip(*combined)

        print('Training . . .')
        sentences_processed = 0
        for src_sentence, tgt_sentence in zip(train_set.source_sentences, train_set.target_sentences):
            loss = encoder_decoder.get_loss(src_sentence, tgt_sentence)
            loss_value = loss.value()
            loss.backward()
            trainer.update()
            sentences_processed += 1
            if sentences_processed % 4000 == 0:
                print('sentences processed: %d' % sentences_processed)

        # Accumulate average losses over training to plot
        val_loss = get_val_set_loss(encoder_decoder, dev_set)
        print('Validation loss this epoch %f' % val_loss)
        losses.append(val_loss)

        print('Translating . . .')
        translated_sentences = encoder_decoder.translate_all(dev_set.source_sentences)

        print('translating {} source sentences...\n'.format(len(sample_output)))
        for sample in sample_output:
            print('Target: {}\nTranslation: {}\n'.format(' '.join(dev_set.target_sentences[sample]),
                                                                         ' '.join(translated_sentences[sample])))

        bleu_score = get_bleu_score(translated_sentences, dev_set.target_sentences)
        print('bleu score: %f' % bleu_score)
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            # save the model
            encoder_decoder.save(model_file)

    print('best bleu score: %f' % best_bleu_score)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = '')
#     parser.add_argument('model_type')
    parser.add_argument('train_src_file')
    parser.add_argument('train_tgt_file')
    parser.add_argument('dev_src_file')
    parser.add_argument('dev_tgt_file')
    parser.add_argument('model_file')
    parser.add_argument('--num_epochs', default = 20, type = int)
    parser.add_argument('--embeddings_init')
    parser.add_argument('--seed', default = 0, type = int)
    parser.add_argument('--dynet-mem')

    args = vars(parser.parse_args())
    args.pop('dynet_mem')

    main(**args)
