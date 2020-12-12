from collections import OrderedDict
from functools import partial
from time import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from bioinfokit.visuz import cluster
from sklearn import manifold, datasets

showcase_t_sne_plot = False


class DimensionReduction:

    def reduce_with_neuron_weights(self, neurons, label, data, base, amount_of_different_labels):
        # reduced neuron weights with t-sne to a 2D result
        reshaped_label = label

        neuron_weights = []
        for neuron in neurons:
            neuron_weights.append(neuron.weights)

        vectors = self.remove_last_array(neuron_weights)

        return self.t_sne(vectors, reshaped_label)

    def t_sne(self, vectors, reshaped_label):
        # calculates the dimension reduction based on t-sne
        t_sne_result = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1).fit_transform(vectors)
        cluster.tsneplot(score=t_sne_result)

        cluster.tsneplot(score=t_sne_result, colorlist=reshaped_label, colordot=(
            '#713e5a', '#63a375', '#edc79b', '#d57a66', '#ca6680', '#395B50', '#92AFD7', '#b0413e', '#4381c1',
            '#736ced'), legendpos='upper right', legendanchor=(1.15, 1))

        return t_sne_result

    def compare_reduction_methods(self, n_points, weights, color, n_neighbors):
        # Next line to silence pyflakes. This import is needed.
        Axes3D

        weights = self.remove_last_array(weights)

        n_components = 2

        fig = plt.figure(figsize=(15, 8))
        fig.suptitle("Manifold Learning with %i points, %i neighbors"
                     % (n_points, n_neighbors), fontsize=14)

        # Add 3d scatter plot
        create_plot_with_all_redutions = False
        if create_plot_with_all_redutions:
            ax = fig.add_subplot(251, projection='3d')
            ax.scatter(weights[:, 0], weights[:, 1], weights[:, 2], c=color, cmap=plt.cm.Spectral)
            ax.view_init(4, -72)

        # Set-up manifold methods
        LLE = partial(manifold.LocallyLinearEmbedding, n_neighbors, n_components, eigen_solver='auto')

        methods = OrderedDict()
        methods['LLE'] = LLE(method='standard')
        methods['LTSA'] = LLE(method='ltsa')
        methods['Hessian LLE'] = LLE(method='hessian')
        methods['Modified LLE'] = LLE(method='modified')
        methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
        methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
        methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                                   n_neighbors=n_neighbors)
        methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                         random_state=0)

        tSNE = 0
        # Plot results
        for i, (label, method) in enumerate(methods.items()):
            t0 = time()
            Y = method.fit_transform(weights)
            if label == "t-SNE":
                tSNE = Y
            t1 = time()

            # print("%s: %.2g sec" % (label, t1 - t0))
            ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
            ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
            ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')

        plt.show()
        plt.close('all')
        return tSNE

    def remove_last_array(self, data):
        new_data = []
        for point in data:
            row = []
            for value in point:
                row.append(value[0])
            new_data.append(row)
        return new_data

