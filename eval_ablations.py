"""Scores ablation runs against HI1. Scoring radii stay (1, 3) for EVERY run,
including the bnd_radius sweep.

Writes the report to a txt file as well as the terminal -- see --out.
"""
import argparse
import datetime
import os
import sys

from evaluation import report

C = "./checkpoints/train_STDC2-Seg-%s/pths/model_maxmIOU75.pth"
H = lambda v='full': dict(use_brh=True, brh_variant=v)

RUNS = [
    ("HI1 full (ref)",      C % "HI1",                 H()),
    ("- BRH (=I1)",         C % "I1",                  dict(use_brh=False)),
    ("- bnd weight (=H1)",  C % "H1",                  H()),
    ("M1 no HR feats",      C % "ABL-M1-noHR",         H('no_hr')),
    ("M2 stride 8",         C % "ABL-M2-s8",           H('s8')),
    ("M3 no residual",      C % "ABL-M3-noRes",        H('no_residual')),
    ("M4 no logit input",   C % "ABL-M4-noLogit",      H('no_logit')),
    ("M5 no gate",          C % "ABL-M5-noGate",       H('no_gate')),
    ("L1 unweighted rank",  C % "ABL-L1-unwRank",      H()),
    ("L2 no OHEM w3",       C % "ABL-L2-noOHEM",       H()),
    ("L2b no OHEM w1",      C % "ABL-L2b-noOHEM-w1",   H()),
    ("L3 no detail loss",   C % "ABL-L3-noDetail",     H()),
    ("L4 no aux heads",     C % "ABL-L4-noAux",        H()),
    ("w=2", C % "SENS-w2", H()), ("w=5", C % "SENS-w5", H()), ("w=8", C % "SENS-w8", H()),
    ("r=1", C % "SENS-r1", H()), ("r=2", C % "SENS-r2", H()), ("r=5", C % "SENS-r5", H()),
]

## the dataset loaders print their split and size on construction; drop those
## from the saved file so it is clean enough to paste straight into notes.
## tqdm writes to stderr, so the progress bars never reach stdout in the first place.
NOISE = ('self.mode', 'self.len')


class Tee(object):
    """Duplicate stdout into a file, line-buffered so NOISE can be filtered."""

    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh
        self.buf = ''

    def write(self, data):
        self.stream.write(data)
        self.buf += data
        while '\n' in self.buf:
            line, self.buf = self.buf.split('\n', 1)
            if not line.startswith(NOISE):
                self.fh.write(line + '\n')

    def flush(self):
        self.stream.flush()
        if self.buf:
            if not self.buf.startswith(NOISE):
                self.fh.write(self.buf)
            self.buf = ''
        self.fh.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='ablation_results.txt',
                    help='txt file to write the report to')
    args = ap.parse_args()

    runs = [r for r in RUNS if os.path.exists(r[1])]
    skipped = [r[0] for r in RUNS if not os.path.exists(r[1])]

    real_stdout = sys.stdout
    with open(args.out, 'w') as fh:
        tee = Tee(real_stdout, fh)
        sys.stdout = tee
        try:
            print("ablation report -- %s" % datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
            print("scale=0.75  backbone=STDCNet1446  brh_mid=64  scoring radii=(1, 3)")
            print("scored %d of %d runs" % (len(runs), len(RUNS)))
            if skipped:
                print("pending (no checkpoint yet): %s" % ", ".join(skipped))
            report(runs, dspth='./data', backbone='STDCNet1446', scale=0.75,
                   use_boundary_8=True, brh_mid=64)
        finally:
            tee.flush()
            sys.stdout = real_stdout

    print("\nreport written to %s" % args.out)
