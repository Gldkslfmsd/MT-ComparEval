from collections import defaultdict, Counter
import os
import pickle

def file_newer_than(f, g):
	"""returns True if file f is newer than file g"""
	return os.path.getctime(f) > os.path.getctime(g)

def file_to_counter(fname, require_pickle=False):
	pickled_counts = "%s.counts.pickled" % fname
	if require_pickle or (os.path.exists(pickled_counts) and
		file_newer_than(pickled_counts, fname)):
#		print("reading pickled %s" % pickled_counts)
		counts,tokens = pickle.load(open(pickled_counts, "rb"))
#		print("reading pickled %s finished" % pickled_counts)
	else:
		counts = Counter()
		tokens = 0
		print("reading %s" % fname)
		with open(fname, "r") as f:
			for line in f:
				toks = line.split()
				counts.update(toks)
				tokens += len(toks)
		print("reading %s finished" % fname)
		pickle.dump((counts, tokens), open(pickled_counts, "wb"))
	return counts, tokens


