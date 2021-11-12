from sklearn.cross_decomposition import CCA
import numpy as np

U = np.random.random_sample(500).reshape(500,1)
V = np.random.random_sample(500).reshape(500,1)

cca = CCA(n_components=1)
cca.fit(U, V)

cca.coef_.shape                   # (5,5)

U_c, V_c = cca.transform(U, V)

U_c.shape                         # (100,1)
V_c.shape                         # (100,1)

U_c, V_c = cca.fit_transform(U, V)

result = np.corrcoef(U_c.T, V_c.T)[0,1]