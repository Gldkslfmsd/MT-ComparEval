from bleu_hook import bleu_wrapper 
from watcher import check_experiment
import os
import sys
import pickle

class MetricTable:
	
	NAME = None

	def score(self, ref, hyp):
		raise NotImplemented()

	def cache_dir(self, exp):
		path = os.path.join("temp", exp.exp_name)
		if not os.path.exists(path):
			os.mkdir(path)
		return path

	def cache_file(self, exp):
		return os.path.join(self.cache_dir(exp), self.NAME + ".pickle")

	def restore_from_cache(self, exp):
		with open(self.cache_file(exp), "rb") as f:
			return pickle.load(f)

	def save_to_cache(self, exp, obj):
		with open(self.cache_file(exp), "wb") as f:
			pickle.dump( obj, f)

	def is_cached(self, exp):
		return os.path.exists(self.cache_file(exp))

	def score_dict(self, exp):
		if self.is_cached(exp):
			return self.restore_from_cache(exp)
		else:
			d = {}
			for n, t in exp.tasks:
				d[n] = self.score(exp.ref_path, t)
			self.save_to_cache(exp, d)
			return d

	def table(self, exp, sort=True, head=True):
		if sort:
			scores = sorted(self.score_dict(exp).items(), key=lambda x: -x[1])
		else:
			scores = sorted(self.score_dict(exp).items(), key=lambda x: x[0])
		len_name = str(max(map(len, list(self.score_dict(exp).keys())+["task"]))+1) 

		tab = []
		if head:
			row = [ ("%"+len_name+"s") % "task", self.NAME ]
			tab.append(row)
		for n,s in scores:
			row = [ ("%"+len_name+"s") % n, "%2.2f" % s ]
			tab.append(row)
		return tab

	def print_latex(self, *a, **kw):
		tab = self.table(*a, **kw)
		print(" \\\\\n".join(" & ".join(row) for row in tab)+" \\\\") 

class MultiColumn(MetricTable):

	def __init__(self, columns):
		self.columns = columns
	
	def table(self, exp, sort=True, head=True):
		"""sorts by the first column"""

		tab = self.columns[0].table(exp, sort, head)
		for next_col in [ col.table(exp, sort, head) for col in self.columns[1:]]:
			for first,n in zip(tab, next_col):
				first.append(n[1])
		return tab


		

class BleuColumn(MetricTable):
	CASED = True

	NAME = "BLEU"

	def score(self, ref, hyp):
		b = bleu_wrapper(ref, hyp, case_sensitive=self.CASED)
		return b*100

class BleuUncasedColumn(BleuColumn):
	CASED = False

	NAME = "BLEU-uncased"

class Experiment:
	
	def __init__(self, exp_path):
		self.exp_name = os.path.basename(exp_path)	
		self.translations_paths, task_dirs = check_experiment(exp_path)	
		self.source_path = os.path.join(exp_path, "source.txt")
		self.ref_path = os.path.join(exp_path, "reference.txt")

		self.task_names = [ os.path.basename(tn) for tn in task_dirs ]

		self.tasks = list(zip(self.task_names, self.translations_paths))



def metrics_table(exp_path):
	translations = check_experiment(exp_path)
	ref_path = os.path.join(exp_path, "reference.txt")


if __name__ == "__main__":
	exp = Experiment(sys.argv[1])
	col = MultiColumn((BleuColumn(), BleuUncasedColumn()))
	col.print_latex(exp)
#	metrics_table(sys.argv[1])
