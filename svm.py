import numpy as np

from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
#import matplotlib.pyplot as plt

import generate_extra_data as gd

import pdb, pickle

class AddBiasTerm(BaseEstimator, TransformerMixin):

	def __init__(self):
		pass

	def fit(self, X, y=None, **kwargs):
		return self

	def transform(self, X, y=None, **kwargs):
		return self.fit_transform(X)

	def fit_transform(self, X, y=None, **kwargs):
		N,M = X.shape
		return np.hstack((np.ones((N,1)), X))

best_degree = 9
best_c = 10
best_gamma = 1e-3

X = np.load("data/train_inputs.npy")[:100,:]
Y = np.load("data/train_outputs.npy")[:100]

Xtrain_full, Xtest, Ytrain_full, Ytest = train_test_split(X,Y, train_size=0.8, random_state=1234)
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain_full, Ytrain_full, train_size=0.8, random_state=1234)


def pick_learner():
	linsvc = ("linsvc", LinearSVC(fit_intercept=False, dual=False) )
	svc = ("svc", SVC() )
	polysvc = ("polysvc", SVC(kernel="poly") )
	scale = ("scale", StandardScaler() )

	N,M = Xtrain.shape
	Xtrain_bias = np.ones((N,1), dtype=float)
	N,M = Xvalid.shape
	Xvalid_bias = np.ones((N,1), dtype=float)
	Xtrain_bias = np.hstack((Xtrain,Xtrain_bias))
	Xvalid_bias = np.hstack((Xvalid,Xvalid_bias))

	svc_params = {"svc__kernel":("rbf","sigmoid")}
	polysvc_params = {"polysvc__degree":list(range(2,11))}
	linsvc_params = {"linsvc__penalty":("l2","l1")}

	scores = []

	p = Pipeline([scale, svc])
	g = GridSearchCV(p, svc_params, n_jobs=-1)
	g.fit(Xtrain_bias, Ytrain)
	score = g.score(Xvalid_bias, Yvalid)
	scores.append(("svc", score, g))

	p = Pipeline([scale, linsvc])
	g = GridSearchCV(p, linsvc_params, n_jobs=-1)
	g.fit(Xtrain_bias, Ytrain)
	score = g.score(Xvalid_bias, Yvalid)
	scores.append(("linsvc", score, g))

	p = Pipeline([scale, polysvc])
	g = GridSearchCV(p, polysvc_params, n_jobs=-1)
	g.fit(Xtrain_bias, Ytrain)
	score = g.score(Xvalid_bias, Yvalid)
	scores.append(("polysvc", score, g))

	joblib.dump(scores, "ignore/learners.pkl")
	return scores


def generate_heat_map():
	svc = ("svc", SVC(kernel="poly", degree=9))
	bias = ("bias", AddBiasTerm())
	scale = ("scale", StandardScaler())
	svc_params = {"svc__C":np.logspace(-2,10,13), 
					"svc__gamma":np.logspace(-9,3,13)}

	
	p = Pipeline([scale, bias, svc])
	g = GridSearchCV(p, svc_params, n_jobs=-1)
	g.fit(Xtrain, Ytrain)
	s = (g, g.score(Xvalid, Yvalid))

	joblib.dump(s, "ignore/heat_map.pkl")
	return g.best_estimator_

def pca_grid_search():
	pca = ("pca", IncrementalPCA())
	scale = ("scale", StandardScaler())
	bias = ("bias", AddBiasTerm())
	svc = ("svc", SVC(kernel="poly",degree=best_degree, C=best_c, gamma=best_gamma))

	with_scale = Pipeline([scale, pca, bias, svc])
	sans_scale = Pipeline([pca, bias, svc])
	d = {}

	pca_params = {"pca__n_components":(10, 50, 100, 200, 500, 1000, 48*48)}
	g = GridSearchCV(with_scale, pca_params, n_jobs=-1)
	g.fit(Xtrain, Ytrain)
	d["with_scale"] = (g, g.score(Xvalid, Yvalid))

	g = GridSearchCV(sans_scale, pca_params, n_jobs=-1)
	g.fit(Xtrain, Ytrain)
	d["sans_scale"] =  (g, g.score(Xvalid, Yvalid))

	joblib.dump(d, "ignore/pca_grid_search.pkl")


def extra_data():
	pca = ("pca", IncrementalPCA(n_components=500))
	scale = ("scale", StandardScaler())
	bias = ("bias", AddBiasTerm())
	svc = ("svc", SVC(kernel="poly", degree=best_degree, C=best_c, gamma=best_gamma))
	p = Pipeline([pca, scale, bias, svc])

	n = 50000
	num_slices = 50
	N,M = Xtrain_full.shape

	scores_perturb = []
	scores_extra = []

	extra_data, extra_targets = gd.generate_extra_data(n)
	perturbed_data, perturbed_targets = gd.perturb_modified_digits(Xtrain_full, Ytrain_full, n)


	for i in range(-1,num_slices):
		Xp = np.vstack((Xtrain_full, extra_data[0:(i+1)*n/num_slices]))
		Yp = np.vstack((Ytrain_full, extra_targets[0:(i+1)*n/num_slices]))
		p.fit(Xp, Yp)
		scores_extra.append({"train_score": p.score(Xp, Yp), "test_score": p.score(Xtest, Ytest),
							"num_examples":N+(i+1)*n/num_slices, "i":i})

		Xp = np.vstack((Xtrain_full, perturbed_data[0:(i+1)*n/num_slices]))
		Yp = np.vstack((Ytrain_full, perturbed_targets[0:(i+1)*n/num_slices]))
		p.fit(Xp, Yp)
		scores_perturb.append({"train_score": p.score(Xp, Yp), "test_score": p.score(Xtest, Ytest),
							"num_examples":N+(i+1)*n/num_slices, "i":i})

	d = {"scores_perturb":scores_perturb, "scores_extra": scores_extra}
	with open("ignore/extra_data_graph.json", "w") as f:
		json.dump(d, f)



def write_to_csv(predictions):
	with open("ignore/out.csv","w") as out:
		out.write("id,prediction\n\r")
		l = []
		for cnt, y in enumerate(predictions):
			l.append("{},{}".format(cnt+1,y))
		out.write("\n\r".join(l))


if __name__ == '__main__':
	pca_grid_search()