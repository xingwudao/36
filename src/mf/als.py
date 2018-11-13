import sys
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve

class ALS(object):
    """
    als matrix factorize algorithm
    """

    def __init__(self, iteration=100,
                 alpha=40, lam=10, factors=10):
        """
        initialize with parameters
        Args:
            iteration: number of computing repeatly
            alpha: parameter about confidence
            lam: regulation parameter
            factors: dimention of user features and item features
        """
        self._iteration = iteration
        self._alpha = alpha
        self._lambda = lam
        self._factor = factors
        self._user_features = None
        self._item_features = None

    def train(self, matrix_data_file):
        """
        train via ALS algorithm on matrix_data_file (sparse matrix data)
        compute the following two lines alternativly:
              x_u = ((Y.T*Y + Y.T*(Cu - I) * Y) + lambda*I)^-1 * (X.T * Cu * p(u))
              y_i = ((X.T*X + X.T*(Ci - I) * X) + lambda*I)^-1 * (Y.T * Ci * p(i))
        Args:
            matrix_data_file: coo format matrix
        returns:
            None
        """
        matrix_data = self._load_matrix(matrix_data_file)
        confidence = matrix_data * self._alpha
        users, items = matrix_data.shape
        X = sparse.csr_matrix(np.random.normal(size = (users, self._factor)))
        Y = sparse.csr_matrix(np.random.normal(size = (items, self._factor)))
        X_I = sparse.eye(users)
        Y_I = sparse.eye(items)
        I = sparse.eye(self._factor)
        lI = self._lambda * I
        for iter_number in range(self._iteration):
            yTy = Y.T.dot(Y)
            xTx = X.T.dot(X)
            for u in range(users):
                u_row = confidence[u,:].toarray()
                p_u = u_row.copy()
                p_u[p_u != 0] = 1.0
                CuI = sparse.diags(u_row, [0])
                Cu = CuI + Y_I
                yT_CuI_y = Y.T.dot(CuI).dot(Y)
                yT_Cu_pu = Y.T.dot(Cu).dot(p_u.T)
                X[u] = spsolve(yTy + yT_CuI_y + lI, yT_Cu_pu)

            for i in range(items):
                i_row = confidence[:,i].T.toarray()
                p_i = i_row.copy()
                p_i[p_i != 0] = 1.0
                CiI = sparse.diags(i_row, [0])
                Ci = CiI + X_I
                xT_CiI_x = X.T.dot(CiI).dot(X)
                xT_Ci_pi = X.T.dot(Ci).dot(p_i.T)
                Y[i] = spsolve(xTx + xT_CiI_x + lI, xT_Ci_pi)
        self._user_features = X
        self._item_features = Y

    def save(self,  vector_data_file):
        sparse.save_npz("%s-user" % vector_data_file, self._user_features)
        sparse.save_npz("%s-item" % vector_data_file, self._item_features)

    def _load_matrix(self, matrix_data_file):
        """
        load data from file, the format of data file is lines as following:
            user item cell
        Args:
            matrix_data_file
        Returns:
            sparse.crs_matrix object
        """
        cells = []
        users = []
        items = []
        with open(matrix_data_file, 'r') as matrix_data:
            for line in matrix_data:
                u,i,c = line.strip().split()
                cells.append(float(c))
                users.append(int(u))
                items.append(int(i))
        return sparse.csr_matrix((cells, (users, items)),
                                 shape=(len(users), len(items)))

if __name__ == "__main__":
    als = ALS()
    als.train(sys.argv[1])
    als.save(sys.argv[2])
