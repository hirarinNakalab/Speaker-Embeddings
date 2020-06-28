import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn.manifold as mf

from hparam import hparam as hp


if __name__ == '__main__':
    args = sys.argv
    input_file = os.path.join(hp.test.simmat_dir, args[1])
    output_file = os.path.join(hp.test.simmat_dir, args[2])
    print("create graph data from", args[1])

    CSVTEXT = np.loadtxt(input_file, delimiter=',', dtype=str)
    LABEL = CSVTEXT[:,0]

    DATATEXT = np.delete(CSVTEXT, 0, 1)

    DATA = DATATEXT.astype(np.float64)
    DATA = DATA * (-1) + 3

    DATANUM = len(DATA)

    mds = mf.MDS(n_components=2, dissimilarity="precomputed", random_state=6)
    pos = mds.fit_transform(DATA)

    SUM=np.sum(pos, axis=0)
    AVE=SUM/DATANUM

    fig = plt.figure()

    plt.scatter(pos[:, 0], pos[:, 1], marker='o')

    #for label
    for label, x, y in zip(LABEL, pos[:, 0], pos[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(70, -20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    plt.draw()
    fig.savefig(output_file)