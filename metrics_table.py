from bleu_hook import bleu_wrapper 
from watcher import check_experiment
from subword_counts import SubwordTextEncoder
import os
import sys
import pickle
import math

from time import gmtime, strftime
from renaming import *

def now():
	return strftime("%Y-%m-%d %H:%M:%S", gmtime())
	
def log(msg):
	print(now(),msg)
	pass

# abstract
class MetricTable:
	
	NAME = None

	recount_tasks = set()

	def score(self, ref, hyp):
		raise NotImplemented()

	def score_log(self, exp, task):
		log("counting %s for %s" % (self.NAME, task.name))
		s = self.score(exp, task)
		log("%s for %s: %2.2f" % (self.NAME, task.name, s))
		return s

	def cache_dir(self, exp):
		path = os.path.join("temp", exp.exp_name)
		if not os.path.exists(path):
			os.mkdir(path)
		return path

	def cache_file(self, exp):
		return os.path.join(self.cache_dir(exp), self.NAME + ".py")

	def restore_from_cache(self, exp):
		with open(self.cache_file(exp), "r") as f:
			return eval(f.read())
#		with open(self.cache_file(exp)[:-2]+"pickle", "rb") as f:
#			return pickle.load(f)


	def save_to_cache(self, exp, d):
		def fl(x):
			if not math.isnan(x):
				return str(x)
			return 'float("nan")'
		with open(self.cache_file(exp), "w") as f:
			print("{",file=f)
			print("\t"+"\n\t".join("'%s':%s," % (k,fl(d[k])) for k in sorted(d.keys())),file=f)
			print("}",file=f)

	def is_cached(self, exp):
#		return True
		return os.path.exists(self.cache_file(exp))

	def score_dict(self, exp):
		if self.is_cached(exp):
			d = self.restore_from_cache(exp)
			for k in list(d.keys()):
				if k not in set(t.name for t in exp.tasks):
#					print("deleting ",k)
					del d[k]
		else:
			d = {}
		for task in exp.tasks:
			t = task.translation_path
			n = task.name
			if n not in d or n in self.recount_tasks:
				d[n] = self.score_log(exp, task)
		self.save_to_cache(exp, d)
		return d

	def format_score(self, score):
		return "%2.2f" % score

	def table(self, exp, sort=True, head=True):
		if sort:
			scores = sorted(self.score_dict(exp).items(), key=lambda x: -x[1])
		else:
			scores = sorted(self.score_dict(exp).items(), key=lambda x: x[0])
		len_name = str(max(map(len, list(map(rename,self.score_dict(exp).keys()))+["run"]))+1) 

		tab = []
		if head:
			row = [ ("%"+len_name+"s") % "task", self.NAME ]
			tab.append(row)
		for n,s in scores:
			n = rename(n)
			row = [ ("%"+len_name+"s") % n, self.format_score(s) ]
			tab.append(row)
		return tab

	def print_latex(self, *a, **kw):
		tab = self.table(*a, **kw)
		print(" \\\\\n".join(" ".join(row) for row in tab)+" \\\\") 

class MultiColumn(MetricTable):

	def __init__(self, columns, sort_by=1):
		self.columns = columns
		self.sort_by = sort_by
	
	def table(self, exp, sort=True, head=True):
		tab = self.columns[0].table(exp, False, head)

		for next_col in [ col.table(exp, False, head) for col in self.columns[1:]]:
			for first,n in zip(tab, next_col):
				first.append(n[1])
		if sort:
			tab = [tab[0]] + sorted(tab[1:], key=lambda x: -float(x[self.sort_by]))
		else:
			tab = [tab[0]] + sorted(tab[1:], key=lambda x: x[0])
		return tab


class BleuColumn(MetricTable):
	CASED = True

	NAME = "BLEU"

	def score(self, exp, task):
		try:
			b = bleu_wrapper(exp.ref_path, task.translation_path, case_sensitive=self.CASED)
		except AssertionError:
			return 0
		return b*100

class BleuUncasedColumn(BleuColumn):
	CASED = False

	NAME = "BLEU-uncased"

class BLEUOther(BleuColumn):
	NAME = "BLEU-other"

	def exclude(self, task):
		if task.name in ["translate_decs_dheads_and_baseline100k",
#			"translate_decs_mult_demo_fbb100",
			"translate_decs_baseline500k",
			"translate_decs_ner_inline",
			"translate_decs_baseline",]:
			return True, 0.0
		return False, None

	def score(self, exp, task):
		e, s = self.exclude(task)
		try:
			if not e:
				b = bleu_wrapper(task.reference_other, task.translation_other, case_sensitive=self.CASED)
				return b*100
			else:
				assert 0
		except AssertionError:
			return float("nan")

class AccuracyOther(BLEUOther):
	
	NAME = "accuracy-other"

	def count_score(self, ref, trans):
		acc = []
		reffile = open(ref, "r")
		transfile = open(trans, "r")
		for rline, pline in zip(reffile, transfile):
			r = rline.split()
			pred = pline.split()
#		if len(r) != len(pred):
#			continue
			c = 0
			t = len(r)
			for rtag, predtag in zip(r+["" for _ in pred], pred):
				if rtag == predtag:
					 c += 1
			acc.append(c/t)
		#print(100*c/t)
		reffile.close()
		transfile.close()
		return 100 * sum(acc)/len(acc)


	def score(self, exp, task):
		e, s = self.exclude(task)
		try:
			if not e:
				a = self.count_score(task.reference_other, task.translation_other)
				return a
			else:
				assert 0
		except AssertionError:
			return float("nan")

import sklearn
class RecallOther(AccuracyOther):
	NAME = "recall-other"

	def score_line(self, ref, pred):
		return sklearn.metrics.recall_score(ref, pred, average="macro")

	def count_score(self, ref, trans):
		print(self.NAME)
		score = []
		reffile = open(ref, "r")
		transfile = open(trans, "r")
		for rline, pline in zip(reffile, transfile):
			pred = pline.split()
			r = rline.split()
			pl = len(pred)
			rl = len(r)
			if pl > rl:
				r += [""]*abs(len(pred)-len(r))
			elif pl < rl:
				pred += [""]*abs(len(pred)-len(r))
			s = self.score_line(r, pred)
			score.append(s)
		#print(100*c/t)
		reffile.close()
		transfile.close()
		return 100 * sum(score)/len(score)

class PrecisionOther(RecallOther):
	NAME = "precision-other"

	def score_line(self, ref, pred):
		return sklearn.metrics.precision_score(ref, pred, average="macro")

class F1Other(RecallOther):
	NAME = "F1-other"

	def score_line(self, ref, pred):
#		r = sklearn.metrics.recall_score(ref, pred, average="macro")
#		p = sklearn.metrics.precision_score(ref, pred, average="macro")
#		f = sklearn.metrics.f1_score(ref, pred, average="macro")
#		print(r,p,f)
		return sklearn.metrics.f1_score(ref, pred, average="macro")



class Step(MetricTable):
	
	NAME = "step"

	def score(self, exp, task):
		return task.step

	def format_score(self, score):
		return "%d" % int(score)

class Separator(MetricTable):

	def __init__(self, separator):
		self.NAME = "separator_%s" % separator
		self.separator = separator

	def score(self, exp, task):
		return 0
	
	def format_score(self, _):
		return self.separator



class Epoch(MetricTable):
	
	NAME = "epoch"

	def score(self, exp, task):
		if task.steps_in_epoch:
			return task.step/task.steps_in_epoch
		return float("nan")

class SubwordCount(MetricTable):

	def __init__(self, side="tgt", on="source", training=None, fun=None):
		self.side = side
		self.on = on
		self.NAME = "subwords_side:%s_on:%s" % (side, on)
		self.training = training
		self.fun = fun


	def select_file(self, exp, task):
		if not exp.has_vocabs():
			log("vocabs are missing")

		if self.on == "source":
			return exp.source_path
		elif self.on == "reference":
			return exp.ref_path
		elif self.on == "training":
			return self.training
		elif self.on == "tag":
			with open("tag","w") as f:
				print("ŽŽŽTranslate", file=f)
			return "tag"
		elif self.on.startswith("other"):
			return self.fun(task)
		return task.translation_path

	def score(self, exp, task):
		name, voc, trans = task.name, task.vocabs[self.side], task.translation_path
		if name not in voc or name not in trans:
			print(name, voc, trans)
		print(name)
		f = self.select_file(exp, task)
		log("counting %s for %s" % (self.NAME, f))
		score = SubwordTextEncoder(voc).avg_subwords_in_file(f)
		log("%s for %s: %2.2f" % (self.NAME, f, score))
		return score

class SharedVocab(MetricTable):
	
	def __init__(self, sourcefile=None, targetfile=None):
		self.sourcefile = sourcefile
		self.targetfile = targetfile
		s = os.path.basename(sourcefile)
		t = os.path.basename(targetfile)
		self.NAME = "sharedvocab_src:%s_tgt:%s" % (s, t)

	def score(self, exp, task):
		log("counting %s for %s and %s" % (self.NAME, self.sourcefile, self.targetfile))
		name, voc = task.name, task.vocabs["tgt"]
		enc = SubwordTextEncoder(voc)
		tar = enc.dict_from_file(self.targetfile)
		src = enc.dict_from_file(self.sourcefile)
		score = 2*len(tar.intersection(src))/(len(tar)+len(src))*100
		log("%s for %s and %s: %2.2f" % (self.NAME, self.sourcefile, self.targetfile, score))
		return score


class Task:
	
	def __init__(self, task_path):
		self.path = task_path

		self.translation_path = os.path.join(task_path, "translation.txt")
		self.source_both = os.path.join(task_path, "source.both.txt")
		self.source_other = os.path.join(task_path, "source.other.txt")
		self.reference_both = os.path.join(task_path, "reference.both.txt")
		self.reference_other = os.path.join(task_path, "reference.other.txt")
		self.translation_other = os.path.join(task_path, "translation.other.txt")

		self.config = os.path.join(task_path, "config.neon")
	
		self.name = os.path.basename(task_path)


		self.vocabs = { 
			"src": os.path.join(task_path, "vocab.src"), 
			"tgt": os.path.join(task_path, "vocab.tgt") }
		self.parse_step()

	def parse_step(self):
		with open(self.config, "r") as f:
			for line in f:
				if "step" in line:
#					print(line)
					try:
						_, _, step, _, ep_steps = line.split()
					except ValueError:
#						print(self.path)
						_, _, step, *_ = line.split()
						ep_steps = "nan"

					self.step = int(step)
					self.steps_in_epoch = float(ep_steps)

	def has_vocabs(self):
		return all(os.path.exists(p) for p in self.vocabs.values())


class Experiment:
	
	def __init__(self, exp_path, exclude=lambda _:False, include_set=[]):
		self.exp_name = os.path.basename(exp_path)	
		translations_paths, task_dirs = check_experiment(exp_path)	
		self.tasks = []
		self.translations_paths = []
		for t,tp in zip(task_dirs, translations_paths):
			if include_set:
				if any(t.endswith(x) for x in include_set):
					self.tasks.append(Task(t))
					self.translations_paths.append(tp)
			else:	
				if not exclude(t):
					self.tasks.append(Task(t))
					self.translations_paths.append(tp)
#				print("including")
			
		self.source_path = os.path.join(exp_path, "source.txt")
		self.ref_path = os.path.join(exp_path, "reference.txt")

	def has_vocabs(self):
		return all(t.has_vocabs() for t in self.tasks)



def metrics_table(exp_path):
	translations = check_experiment(exp_path)
	ref_path = os.path.join(exp_path, "reference.txt")


def big_BLEU_voc():
	exclude = lambda x: any(x.endswith(y) for y in ["translate_decs_vf_mult_pos_small","translate_decs_vf_mult_pos_full"])
	print(exclude("translate_decs_vf_mult_pos_small"))
#	exclude = lambda x: True
	exp = Experiment(sys.argv[1], exclude=exclude)
	print(exp.tasks)
	MetricTable.recount_tasks = set([
#		'translate_decs_mult_repeat',
##		'translate_decs_vf_mult_dheads',
#		'translate_decs_vff_mult_pos_full',
#		'translate_decs_vff_mult_pos_small',
#		'translate_decs_vff_mult_dtags',
		])

	amp = Separator("&")
	br = Separator('{}')
	columns = [ 
		amp,
		BleuColumn(), 
		amp, 
		BLEUOther(),
		amp, 
		Step(),
		amp,
		Epoch(),
		amp,
#		BleuUncasedColumn(),
		SubwordCount(side="tgt",on="translation"),
		br,
		SubwordCount(side="tgt",on="reference"), 
		br,
		SubwordCount(side="tgt",on="training",training="mtce-data/cs100k"), 
		amp,

		SubwordCount(side="src",on="source"), 
		br,
		SubwordCount(side="src",on="training",training="mtce-data/de100k"), 
		amp,

#		SubwordCount(side="src",on="tag"), 

#		BLEUOther(),
		SubwordCount(side="tgt",on="other-trans", fun=lambda t: t.translation_other),
		br,
		SubwordCount(side="tgt",on="other-ref", fun=lambda t: t.reference_other)
#		SharedVocab(sourcefile="mtce-data/dev-600000/source.txt",targetfile="mtce-data/dev-600000/reference.txt")
		]
	col = MultiColumn(columns, 2)
	col.print_latex(exp)


def accuracy_etc():
	include_set = ADEQ
#.	include_set.add('translate_decs_mult_dht')
#.	print(include_set)
	exp = Experiment(sys.argv[1], include_set=include_set)
	amp = Separator("&")
	br = Separator('{}')
	columns = [ 
	
		amp,
		BleuColumn(), 
		amp, 
		BLEUOther(),
		amp,
#		AccuracyOther(),
#		amp, 
		RecallOther(),
		amp, 
		PrecisionOther(),
		amp,
		F1Other(),
		amp,

		Step(),
		amp,
		Epoch(),
	]
	col = MultiColumn(columns, 2)
	col.print_latex(exp)

def sampling(data=SAMPLING):
	include_set = data 
#.	include_set.add('translate_decs_mult_dht')
#.	print(include_set)
	exp = Experiment(sys.argv[1], include_set=include_set)
	amp = Separator("&")
	br = Separator('{}')
	columns = [ 
	
		amp,
		BleuColumn(), 
		amp, 
		BLEUOther(),
		amp,
#		AccuracyOther(),
#		amp, 
		RecallOther(),
		amp, 
		PrecisionOther(),
		amp,
		F1Other(),
		amp,

		Step(),
		amp,
		Epoch(),
	]
	col = MultiColumn(columns, 2)
	col.print_latex(exp)

def ne_table():
	d = ne_tasks()
	include_set = d
#.	include_set.add('translate_decs_mult_dht')
#.	print(include_set)
	exp = Experiment(sys.argv[1], include_set=include_set)
	amp = Separator("&")
	br = Separator('{}')
	columns = [ 
	
		amp,
		BleuColumn(), 
		amp, 
		BLEUOther(),
		amp,
#		AccuracyOther(),
#		amp, 
	#	RecallOther(),
	#	amp, 
	#	PrecisionOther(),
	#	amp,
		AccuracyOther(),
#		F1Other(),
		amp,

		Step(),
		amp,
		Epoch(),
	]
	col = MultiColumn(columns, 2)
	col.print_latex(exp)


def tsd_table():
#	d = ne_tasks()
#	include_set = d
#.	include_set.add('translate_decs_mult_dht')
#.	print(include_set)
	exp = Experiment(sys.argv[1])
	amp = Separator("&")
	br = Separator('{}')
	columns = [ 
	
		amp,
		BleuColumn(), 
#		AccuracyOther(),
#		amp, 
	#	RecallOther(),
	#	amp, 
	#	PrecisionOther(),
	#	amp,
#		F1Other(),
	]
	col = MultiColumn(columns, 2)
	col.print_latex(exp)




if __name__ == "__main__":
#	big_BLEU_voc()
#	accuracy_etc()
#	sampling()

	

	ne_table()
#	tsd_table()

