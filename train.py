import numpy as np, matplotlib.pyplot as plt, os, argparse, time
from pathlib import Path
import char_rnn, data_utils


SEQ_LEN   = 25
HIDDEN    = 100
ETA       = 1e-3
EPOCHS    = 2                 
PRINT_EVERY   = 100
SAMPLE_EVERY  = 1_000
SAMPLE_LEN    = 200

MAX_UPDATES  = 100_000    
PRE_CHAR     = '.'       
BEST_SAMPLE_LEN = 1000    

def ensure_out():
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    return out


def train(book_path: str):
    out = ensure_out()
    book = data_utils.read_book(book_path)
    chars, K, c2i, i2c = data_utils.build_vocab(book)

    rnn = char_rnn.CharRNN(K, HIDDEN, ETA)
    smooth_loss = None
    losses, iters = [], []

    seed_x = data_utils.chars_to_onehot(PRE_CHAR, c2i)[:, :1]
    pre_txt = data_utils.onehot_to_string(
        rnn.sample(np.zeros((HIDDEN, 1)), seed_x, SAMPLE_LEN), i2c
    )
    (out / "synth_iter000000.txt").write_text(pre_txt, encoding="utf-8")

    # --- SGD loop ---
    e = 0
    hprev = np.zeros((HIDDEN,1))
    N = len(book) - SEQ_LEN - 1
    total_updates = int(EPOCHS * N / SEQ_LEN)
    t0 = time.time()

    for it in range(1, MAX_UPDATES + 1):
        if e + SEQ_LEN + 1 >= len(book):
            e = 0
            hprev = np.zeros_like(hprev)     

        xchars = book[e:e+SEQ_LEN]
        ychars = book[e+1:e+SEQ_LEN+1]
        X = data_utils.chars_to_onehot(xchars, c2i)
        Y = data_utils.chars_to_onehot(ychars, c2i)

        loss, hprev = rnn.train_step(X, Y, hprev)
        smooth_loss = loss if smooth_loss is None else 0.999*smooth_loss + 0.001*loss

        # ----- bookkeeping -----
        if it % PRINT_EVERY == 0:
            print(f"[{it:>7}/{MAX_UPDATES}] smooth_loss={smooth_loss: .4f}")

        if it % SAMPLE_EVERY == 0 or it == 1:
            seed_x = X[:, :1]    # first char in current mini-seq
            Y = rnn.sample(hprev, seed_x, SAMPLE_LEN)
            sample_txt = data_utils.onehot_to_string(Y, i2c)
            fname = out / f"synth_iter{it:06d}.txt"
            fname.write_text(sample_txt, encoding='utf-8')

        if it % PRINT_EVERY == 0:
            losses.append(smooth_loss)
            iters.append(it)

        # best model checkpoint
        if smooth_loss == min(losses, default=smooth_loss):
            np.savez(out / "best_model.npz", **rnn.params)

        e += SEQ_LEN

    # --- plot ---
    plt.figure()
    plt.plot(iters, losses)
    plt.xlabel("update iteration")
    plt.ylabel("smooth loss")
    plt.title("Training curve – CharRNN on *Goblet of Fire*")
    plt.tight_layout()
    plt.savefig(out / "smooth_loss.png", dpi=180)

    best = np.load(out / "best_model.npz")
    rnn.params = {k: best[k] for k in best.files}          # load weights
    seed_x = data_utils.chars_to_onehot(PRE_CHAR, c2i)[:, :1]
    best_txt = data_utils.onehot_to_string(
        rnn.sample(np.zeros((HIDDEN, 1)), seed_x, BEST_SAMPLE_LEN), i2c
    )
    (out / "best_sample_1000.txt").write_text(best_txt, encoding="utf-8")

    dt = time.time() - t0
    print(f"\nDone in {dt/60: .1f} min.  All artefacts written to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=str, default="goblet_book.txt",
                        help="Path to goblet_book.txt")
    args = parser.parse_args()
    train(args.book)
