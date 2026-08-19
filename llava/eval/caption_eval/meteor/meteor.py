#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import subprocess
import threading
import shutil
from pathlib import Path

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR
JAVA_BIN='java'


def _ascii_runtime_dir():
    """Run METEOR outside the Chinese project path.

    METEOR 1.5 mis-encodes non-ASCII paths while locating its paraphrase
    database, so it must use an ASCII working directory.
    """
    source_dir = Path(__file__).resolve().parent
    runtime_dir = Path('/tmp/3drs_vips_meteor')
    runtime_dir.mkdir(parents=True, exist_ok=True)
    for name in (METEOR_JAR, 'data'):
        source = source_dir / name
        target = runtime_dir / name
        if target.is_symlink():
            target.unlink()
        if source.is_dir():
            if not target.exists():
                shutil.copytree(source, target)
        elif not target.exists() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
    return runtime_dir

class Meteor:

    def __init__(self):
        runtime_dir = _ascii_runtime_dir()
        self.meteor_cmd = [JAVA_BIN, '-Xmx2G', '-jar', METEOR_JAR, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=str(runtime_dir), \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        # Used to guarantee thread safety
        self.lock = threading.Lock()

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        self.lock.acquire()
        for i in imgIds:
            assert(len(res[i]) == 1)
            stat = self._stat(res[i][0], gts[i])
            eval_line += ' ||| {}'.format(stat)

        self.meteor_p.stdin.write('{}\n'.format(eval_line).encode())
        self.meteor_p.stdin.flush()
        for i in range(0,len(imgIds)):
            scores.append(float(self.meteor_p.stdout.readline().strip()))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
        self.meteor_p.stdin.flush()
        return self.meteor_p.stdout.readline().decode().strip()

    def _score(self, hypothesis_str, reference_list):
        self.lock.acquire()
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||','').replace('  ',' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write('{}\n'.format(score_line))
        stats = self.meteor_p.stdout.readline().strip()
        eval_line = 'EVAL ||| {}'.format(stats)
        # EVAL ||| stats 
        self.meteor_p.stdin.write('{}\n'.format(eval_line))
        score = float(self.meteor_p.stdout.readline().strip())
        self.lock.release()
        return score
 
    def __exit__(self):
        self.lock.acquire()
        self.meteor_p.stdin.close()
        self.meteor_p.wait()
        self.lock.release()
