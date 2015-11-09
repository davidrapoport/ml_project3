from sklearn.externals import joblib
from matplotlib import pyplot as plt
from svm import AddBiasTerm
import numpy as np
import json

heat_map, hm_score = joblib.load("ignore/heat_map.pkl")

g = "svc__gamma"
c = "svc__C"

cvals = heat_map.param_grid[c]
gvals = heat_map.param_grid[g]
heatmap_matrix = np.zeros((len(heat_map.param_grid[g]), len(heat_map.param_grid[c])))

c_grid = dict([(val, i) for i, val in enumerate(heat_map.param_grid[c])])
g_grid = dict([(val, i) for i, val in enumerate(heat_map.param_grid[g])])

for score in heat_map.grid_scores_:
	heatmap_matrix[g_grid[score.parameters[g]], c_grid[score.parameters[c]]] = score.mean_validation_score


def heatmap(model, xvals, yvals, scores, x_name, y_name, name):
	fig = plt.figure()
	plt.imshow(scores, interpolation='nearest', aspect='auto')

	title = 'Accuracy for ' + model
	plt.title(title)
	plt.xticks(range(len(xvals)), xvals)
	plt.yticks(range(len(yvals)), yvals)
	plt.xlabel(x_name)
	plt.ylabel(y_name)

	plt.jet()
	plt.colorbar()

	# plt.show()
	#if not os.path.exists(plot_dir):
    #        os.makedirs(plot_dir)

	fig.savefig(name,format='png', bbox_='')
heatmap("SVM", cvals, gvals, heatmap_matrix, "C", "Gamma", "SVM_grid_search.png")

svc, linsvc, polysvc = joblib.load("ignore/learners.pkl")
d = {}
d["linsvc"] = linsvc[1]
for degree in polysvc[2].param_grid['polysvc__degree']:
	m = max([g.mean_validation_score for g in polysvc[2].grid_scores_ if g.parameters['polysvc__degree']== degree])
	d["polysvc_"+str(degree)] = m

for kernel in svc[2].param_grid['svc__kernel']:
	m = max([g.mean_validation_score for g in svc[2].grid_scores_ if g.parameters['svc__kernel']== kernel])
	d["svc_"+str(kernel)] = m

with open("learner_results.json", "w") as f:
	json.dump(d,f)

pca_search = joblib.load("ignore/pca_grid_search.pkl")
xvals = ["Centered", "Not Centered"]
scale, _ = pca_search['with_scale']
sans, _ = pca_search['sans_scale']
heatmap_matrix = np.zeros((len(sans.param_grid['pca__n_components']),2))
grid = dict([(val, i) for i, val in enumerate(sans.param_grid['pca__n_components'])])
for score in scale.grid_scores_:
	heatmap_matrix[grid[score.parameters['pca__n_components']], 0] = score.mean_validation_score

for score in sans.grid_scores_:
	heatmap_matrix[grid[score.parameters['pca__n_components']], 1] = score.mean_validation_score

heatmap("PCA selection", xvals, sans.param_grid['pca__n_components'], heatmap_matrix, 
		"Scaled", "Number Components", "PCA_gridsearch.png")
