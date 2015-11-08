import numpy as np

from sklearn.decomposition import PCA,IncrementalPCA
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib.pyplot as plt

import generate_extra_data as gd

import pdb, pickle

X = np.load("data/train_inputs.npy")#[:100,:]
Y = np.load("data/train_outputs.npy")#[:100]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, train_size=0.8, random_state=1234)
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, train_size=0.8, random_state=1234)


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

	svc_params = {"svc__kernel":("rbf","sigmoid"), "svc__C":(0.1,1.0,10.0,100.0,1000.0), "svc__gamma":(1e-3,1e-4,1e-5,1e-2)
					}
	polysvc_params = {"polysvc__degree":(3,5,9), "polysvc__C":(0.1,1.0,10.,100.,1000.), "polysvc__gamma":(1e-3,1e-4,1e-5,1e-2)}
	linsvc_params = {"linsvc__penalty":("l2","l1"),"linsvc__C":(0.1,1.0,10.,1000.)}

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

pick_learner()

def generate_svm_benchmarks():

	linsvc = LinearSVC(fit_intercept=False, dual=False)
	svc = SVC()
	polysvc = SVC(kernel="poly")
	scale = StandardScaler()
	no_scale = StandardScaler(with_mean=False, with_std=False)

	num_extra=500
	extra_data, extra_targets = gd.generate_extra_data(n=num_extra, elastic=True)
	more_data, more_targets = gd.perturb_modified_digits(X,Y,n=num_extra)
	bias = np.ones((num_extra,1))
	extra_data = np.hstack((bias, extra_data))
	more_data = np.hstack((bias, more_data))

	X = np.hstack((np.ones((N,1)),X))
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, train_size=0.8, random_state=1234)
	test = np.load("data/test_inputs.npy")
	test = np.hstack((np.ones((test.shape[0],1)), test))


	Xtrain_extra = np.vstack((Xtrain, extra_data))
	Xtrain_affine = np.vstack((Xtrain, more_data))
	Ytrain_extra = np.append(Ytrain, extra_targets)
	Ytrain_affine = np.append(Ytrain, more_targets)

	pca100 = IncrementalPCA(n_components=100)
	pca200 = IncrementalPCA(n_components=200)
	pca500 = IncrementalPCA(n_components=500)
	pca1000 = IncrementalPCA(n_components=1000)
	pcafull = IncrementalPCA(n_components=M+1)

	datasets = [ ("normal", (Xtrain, Ytrain))] #, ("extra_generated", (Xtrain_extra, Ytrain_extra)), 
				#("extra_affine", (Xtrain_affine, Ytrain_affine))

	preprocessers = [("scale", scale)]


	linsvc_params = {"linsvc__penalty":("l2","l1"),"linsvc__C":(1.0,2.0,5.0)}
	svc_params = {"svc__C":(1.0,2.0,5.0), "svc__kernel":("rbf","sigmoid")}
	polysvc_params =	{"polysvc__C":(1.0,2.0,5.0), "polysvc__degree":(3, 5, 9)}

	pcas = [ ("pcafull", pcafull), ("pca100", pca100), ("pca200", pca200), ("pca500", pca500), ("pca1000", pca1000)]
	learners = [("linsvc", linsvc, linsvc_params), ("svc", svc, svc_params), ("polysvc", polysvc, polysvc_params)]

	scores = []
	for data_name, data in datasets:
		for pca_name, pca in pcas:
			for preproc in preprocessers:
				for learner in learners:
					print data_name+" "+preproc[0]+" "+ pca_name +" "+learner[0]
					p = Pipeline([preproc, (pca_name, pca), (learner[0], learner[1])])
					g = GridSearchCV(p, learner[2], n_jobs=1)
					g.fit(*data)
					scores.append({"name":data_name+" "+preproc[0]+" "+ pca_name +" "+learner[0], 
									"grid_scores": g.grid_scores_, "score":g.score(Xtest, Ytest),
									"prediction":g.predict(test), "grid_search":g})
					
	for score in scores:
		print "{}: {}".format(score['name'], score['score'])
	joblib.dump(scores, "ignore/benchmark.pkl")
	scores.sort(key=lambda x:-x['score'])
	write_to_csv(scores[0]['prediction'])










def write_to_csv(predictions):
	with open("ignore/out.csv","w") as out:
		out.write("id,prediction\n\r")
		l = []
		for cnt, y in enumerate(predictions):
			l.append("{},{}".format(cnt+1,y))
		out.write("\n\r".join(l))

