import os
import sys
import math
import operator
import numpy as np
import pylab as pl
import cPickle as pickle
import matplotlib.pyplot as plt

svm_c_vals = [0.01,0.1,1.0,10.0,100.0]
svm_gamma_vals = [0.01,0.1,1.0,10.0,100.0]
ada_maxd_vals = [0.5,1,2,4]
ada_nest_vals = [100,300,600,1200]

all_scores = {'SVM':[[0.44211,0.54956,0.50040,0.43585,0.43585],[0.54528,0.58187,0.55722,0.51045,0.51045],
					[0.57389,0.59019,0.57652,0.55083,0.55083],[0.58533,0.58915,0.57193,0.55083,0.55083],
					[0.58737,0.57863,0.57193,0.55083,0.55083]],
					'ADA':[[0.58074,0.57951,0.57846,0.57702],[0.57917,0.57769,0.57642,0.57600],
					[0.46617,0.46617,0.46617,0.46617],[0.32281,0.32281,0.32281,0.32281]]}
param_names = {'SVM':['C','Gamma'],'ADA':['Learning Rate','N Estimators']}
param_vals = {'SVM':{'C':svm_c_vals,'Gamma':svm_gamma_vals},'ADA':{'Learning Rate':ada_maxd_vals,'N Estimators':ada_nest_vals}}

score_fn = 'Accuracy'
directory = './Scores'

def heatmap(scores,p1,p2,pvals,model,score_fn):
	fig = plt.figure()
	plt.imshow(scores, interpolation='nearest', aspect='auto')

	title = ' '.join([score_fn,'for',model])
	plt.title(title)
	p1_vals = pvals[p1]
	p2_vals = pvals[p2]
	plt.xticks(range(len(p1_vals)), p1_vals)
	plt.yticks(range(len(p2_vals)), p2_vals)
	plt.xlabel(p1)
	plt.ylabel(p2)

	plt.jet()
	plt.colorbar()

	#if not os.path.exists(plot_dir):
    #        os.makedirs(plot_dir)

	plot_path = directory + '/' + title + '.pdf'
	print plot_path
	fig.savefig(plot_path,format='png', bbox_='')

if __name__ == '__main__':
	for model,scores in all_scores.iteritems():
		print model
		p1_name = param_names[model][0]
		p2_name = param_names[model][1]
		params = param_vals[model]
		for i in range(len(scores)):
			for j in range(len(scores[0])):
				print p1_name, params[p1_name][i],',',p2_name, params[p2_name][j],":"
				print scores[i][j]
		heatmap(scores, p1_name, p2_name, params, model, score_fn)

