import sqlite3
import os
import sys
from watcher_metrics import * 
from utils import escape


def log(msg):
#	print(msg, file=sys.stderr)
	pass

def check_task(task_path):
	log("checking task %s" % task_path)
	files = os.listdir(task_path)
	assert "translation.txt" in files

def check_experiment(exp_dirname):
	log("checking experiment in %s" % exp_dirname)
	files = os.listdir(exp_dirname)
	assert "source.txt" in files, "source.txt is missing"
	assert "reference.txt" in files, "reference.txt is missing"

	at_least_one_task = False

	check_len = [ os.path.join(exp_dirname, "source.txt"), os.path.join(exp_dirname, "reference.txt") ]
	task_dirs = []
	for task in files:
		if task in [ "source.txt", "reference.txt" ]:
			continue
		task_path = os.path.join(exp_dirname, task)
		if os.path.isdir(task_path):
			try:
				check_task(task_path)
				task_dirs.append(task_path)
				check_len.append(os.path.join(task_path, "translation.txt"))
			except AssertionError:
				log("invalid task %s" % task_path)
			else:
				at_least_one_task = True
		else:
			log("found extra file %s" % task_path)
	tasks = []
	if at_least_one_task:
		with open(check_len[0],"r") as f:
			l = len(f.readlines())
		log("file %s has %d lines" % (check_len[0], l))
		for ch in check_len[1:]:
			with open(ch, "r") as f:
				cl = len(f.readlines())
				log("file %s has %d lines" % (ch, cl))
				if l != cl:
					raise ValueError("Incorrect number of lines in file %s" % ch)
			tasks.append(ch)
	else:
		raise ValueError("no task present")

	log("OK, experiment %s seems to have valid format" % exp_dirname)
	return tasks[1:], task_dirs

def import_experiment(exp_dirname):
	tasks = check_experiment(exp_dirname)
	print(exp_dirname)
	if exp_dirname.endswith("/"):
		exp_dirname = exp_dirname[:-1]
	exp_name = os.path.basename(exp_dirname)
	print(exp_name)
	i = 0
#	while
	n = exp_name
	while True:
		c = con.cursor()
		s = c.execute("select id from experiments where name = '%s'" % (n))
		if not s.fetchall():
			break
		i += 1
		n = exp_name + ("-%s" % i)
	exp_name = n
	c = con.cursor()
	print("insert into experiments (name, url_key, description, visible) values ('%s', '%s', '', 1);" % (exp_name, exp_name))
	c.execute("insert into experiments (name, url_key, description, visible) values ('%s', '%s', '', 1);" % (exp_name, exp_name))
	con.commit()

	s = c.execute("select id from experiments where name = '%s'" % (exp_name))
	exp_id, *_ = s.fetchone()

	source = open(os.path.join(exp_dirname, "source.txt"), "r")
	ref_path = os.path.join(exp_dirname, "reference.txt")
	reference = open(ref_path, "r")
	c = con.cursor()
	i = 1
	for s,r in zip(source, reference):
		s = escape(s)
		r = escape(r)
#		print(i,"""insert into sentences (experiments_id, source, reference) values (%d, '%s', '%s');""" % (exp_id, s,r))
		c.execute("""insert into sentences (experiments_id, source, reference) values (%d, '%s', '%s');""" % (exp_id, s,r))
		i += 1
	con.commit()
	log("sentences inserted")

	for t in tasks:
		import_task(t, exp_id, ref_path)




def import_task(task_path, exp_id, ref_path):
	task_name = os.path.basename(task_path.replace("/translation.txt",""))
	print(task_path, task_name)

	n = task_name
	c = con.cursor()
	while True:
		s = c.execute("select id from tasks where name = '%s' and experiments_id = '%d'" % (n,exp_id))
		if not s.fetchall():
			break
		i += 1
		n = exp_name + ("-%s" % i)
	task_name = n
	cmd = "insert into tasks (experiments_id, name, url_key, description, visible) values ('%d', '%s', '%s', '', 1);" % (exp_id, task_name, task_name)
	c.execute(cmd)
	con.commit()

	c = con.cursor()
	s = c.execute("select id from tasks where name = '%s' and experiments_id = %d;" % (task_name, exp_id))
	task_id, *_ = s.fetchone()

	print(task_id)

	
	metrics = [ BleuMetric, BleuCasedMetric ]
	for m in metrics:
		m = m(con)
		m.task_metric(task_id, task_path, ref_path)
	
#	InsertSentences(con).insert_sentences(task_id, task_path, ref_path)







if __name__ == "__main__":

	con = sqlite3.connect("storage/database")
#check_experiment(sys.argv[1])
	x = import_experiment(sys.argv[1])

	con.close()
