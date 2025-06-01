import pathlib, numpy as np, data_utils, char_rnn

book = pathlib.Path('goblet_book.txt').read_text()

_, K, c2i, i2c = data_utils.build_vocab(book)

rnn = char_rnn.CharRNN(K)                     

seed = data_utils.chars_to_onehot('.', c2i)[:, :1]
txt  = data_utils.onehot_to_string(rnn.sample(np.zeros((rnn.m,1)), seed, 200), i2c)

pathlib.Path('outputs/synth_iter000000.txt').write_text(txt)
print('wrote outputs/synth_iter000000.txt')
