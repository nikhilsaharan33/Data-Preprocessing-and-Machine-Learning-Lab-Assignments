import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import statistics as st

df = pd.read_csv('pima-indians-diabetes.csv')

#1

# dropping attribute 'class'
df.drop(["class"], axis = 1, inplace=True)
# replacing outliers with median
col_names = df.columns.values.tolist()
for i in col_names:
    Q1 = np.percentile(df[i], 25, interpolation='midpoint')
    Q3 = np.percentile(df[i], 75, interpolation='midpoint')
    IQR = Q3-Q1 # inter-quartile range
    median = df[i].median()
    # replacing outliers with median
    df.loc[df[i]<Q1-(1.5 * IQR),i] = median
    df.loc[df[i]>Q3+(1.5 * IQR),i] = median

# (a)
# Min-Max normalization
print("Q 1 (a):")
before = {}
after = {}
df_copy = df.copy()
for i in col_names:
    minm = min(df_copy[i])
    maxm = max(df_copy[i])
    before[i] = [minm, maxm]
    new_minm = 5
    new_maxm = 12
    old = df_copy[i].values.tolist()
    new = []
    for value in old:
        x = ((value - minm) * (new_maxm - new_minm) / (maxm - minm)) + new_minm
        new.append(x)

    df_copy[i] = df_copy[i].replace(old, new)
    after[i] = [min(df_copy[i]), max(df_copy[i])]

print("The minimum and maximum values before performing the Min-Max normalization of the attributes is\n", before)
print("The minimum and maximum values after performing the Min-Max normalization of the attributes is\n", after)

# (b)
# Standardization
print("Q 1 (b):")
before_mean = {}
before_std = {}
after_mean = {}
after_std = {}
for i in col_names:
    mean = df[i].mean()
    std = df[i].std()
    before_mean[i] = round(mean, 3)
    before_std[i] = round(std, 3)
    old = df[i].values.tolist()
    new = []
    for value in old:
        x = (value - mean) / std
        new.append(x)

    df[i] = df[i].replace(old, new)
    after_mean[i] = round(df[i].mean(), 3)
    after_std[i] = round(df[i].std(), 3)

print("The mean before standardization is\n", before_mean)
print("The standard deviation before standardization is\n", before_std)
print("The mean after standardization is\n", after_mean)
print("The standard deviation after standardization is\n", after_std)

# 2
# generating 2-dimensional synthetic data of 1000 samples
mean_2 = [0, 0]
cov = [[13, -3], [-3, 5]]
data = np.random.multivariate_normal(mean_2, cov, 1000)
df0 = pd.DataFrame(data, columns=['X1', 'X2'])

# (a)
# scatter plot of the data samples
plt.scatter(df0['X1'], df0['X2'])
plt.title("Scatter plot of data samples")
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# (b)
# computing the eigenvalues and eigenvectors of the covariance matrix
print("Q 2 (b):")
w, v = np.linalg.eig(cov)
print("The eigen values are", w)
print("The eigen vectors are", v)

# plotting the Eigen directions (with arrows/lines) onto the scatter plot of data
plt.scatter(df0['X1'], df0['X2'], marker='x')
plt.quiver(v[0][0], v[0][1], scale=4)
plt.quiver(v[1][0], v[1][1], scale=4)
plt.title('Plot of 2D Synthetic Data and Eigen directions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# (c)
# First Eigen Direction
unit1 = [v[0][0] / ((v[0][0]) ** 2 + (v[0][1]) ** 2) ** (1 / 2),
         v[0][1] / ((v[0][0]) ** 2 + (v[0][1]) ** 2) ** (1 / 2)]  # unit vector in direction of first eigen vector
sum1 = df0['X1'] * unit1[0] + df0['X2'] * unit1[1]
df0['sum1'] = sum1
# Projecting the data on to the first eigen direction
e1x = df0['sum1'] * unit1[0]
e1y = df0['sum1'] * unit1[1]
e1x = [round(num, 3) for num in e1x.tolist()]
e1y = [round(num, 3) for num in e1y.tolist()]
# superimposed scatter plots with eigen directions
plt.scatter(df0['X1'], df0['X2'], marker='x')
plt.quiver(v[0][0], v[0][1], scale=4)
plt.quiver(v[1][0], v[1][1], scale=4)
plt.scatter(e1x, e1y, marker='x')
plt.title('Projected values onto the first eigen directions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Second Eigen Direction
unit2 = [v[1][0] / ((v[1][0]) ** 2 + (v[1][1]) ** 2) ** (1 / 2),
         v[1][1] / ((v[1][0]) ** 2 + (v[1][1]) ** 2) ** (1 / 2)]  # unit vector in direction of second eigen vector
sum2 = df0['X1'] * unit2[0] + df0['X2'] * unit2[1]
df0['sum2'] = sum2
# Projecting the data on to the second eigen direction
e2x = df0['sum2'] * unit2[0]
e2y = df0['sum2'] * unit2[1]
e2x = [round(num, 3) for num in e2x.tolist()]
e2y = [round(num, 3) for num in e2y.tolist()]
# superimposed scatter plots with eigen directions
plt.scatter(df0['X1'], df0['X2'], marker='x')
plt.quiver(v[0][0], v[0][1], scale=4)
plt.quiver(v[1][0], v[1][1], scale=4)
plt.scatter(e2x, e2y, marker='x')
plt.title('Projected values onto the second eigen direction')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# (d)
# reconstructing the data sample using both eigen vectors
print("Q 2 (d):")
df0 = df0.drop(['sum2', 'sum1'], axis=1)
pca = PCA(n_components=2)
dfx = pca.inverse_transform(pca.fit_transform(df0))
mse = np.linalg.norm((dfx - df0), None)
print("The Reconstruction error between new and original matrix is", mse)

# 3
df_eigen = df.copy()
# Subtracting mean
for i in col_names:
    mean = df_eigen[i].mean()
    original = df_eigen[i].values.tolist()
    mean_subtracted = []
    for value in original:
        mean_subtracted.append(value - mean)

    df_eigen[i] = df_eigen[i].replace(original, mean_subtracted)

# Computing Correlation Matrix
corr_matrix = df_eigen.corr()

# Performing Eigen Analysis
val, vec = np.linalg.eig(corr_matrix.to_numpy())
eigen = {}
for i in range(len(val)):
    eigen[round(val[i], 3)] = [round(num, 3) for num in vec[i]]
# arranging the eigen vectors in descending order of their respective eigen values
sorted_eigen = sorted(eigen.items(), reverse=True)
eigen_analysis = {}
for i in range(len(val)):
    eigen_analysis[round(sorted_eigen[i][0], 3)] = [round(num, 3) for num in sorted_eigen[i][1]]

# (a)
# Reducing the multidimensional (d = 8) data into lower dimensions (l = 2)
print("Q 3 (a):")
pca = PCA(n_components=2)
reduced = pca.fit_transform(df)
new_df = pd.DataFrame(reduced, columns=['x1', 'x2'])
print(new_df)
# calculating variances
var1 = st.variance(new_df['x1'].values.tolist())
var2 = st.variance(new_df['x2'].values.tolist())
print("The Variances are", round(var1, 3), "and", round(var2, 3))
print("The Eigen Values are", round(sorted_eigen[0][0], 3), "and", round(sorted_eigen[1][0], 3))
# Scatter plot of reduced dimensional data with l=2
plt.scatter(new_df['x1'], new_df['x2'], marker='x')
plt.title('Scatter plot of reduced dimensional data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# (b)
# Plotting all the eigenvalues in the descending order
n = np.linspace(1, 8, 8)
plt.plot(n, eigen_analysis.keys())
plt.title('Eigen Values in descending order')
plt.xlabel('Position')
plt.ylabel('Eigen Values')
plt.show()

# (c)
col = ['x1','x2','x3','x4','x5','x6','x7','x8']
# calculating the  reconstruction errors in terms of RMSE considering the different values of l (=1, 2, ..., 8)
print("Q 3 (c):")
components = [i for i in range()]
RMSE = []
for n in components:
    pca = PCA(n_components=n)
    reduced_to_l = pca.fit_transform(df)
    dfn = pca.inverse_transform(pca.fit_transform(df))
    # printing the covariance matrix of each of the l-dimensional representations (l = 2, 3, ..., 8)
    if n != 1:
        cols = col[0:n]
        dfl = pd.DataFrame(data=reduced_to_l, columns=[cols])
        print("The covariance matrix for value of l =", n, ":\n", dfl.cov().round(3))

    rmse = np.linalg.norm((df - dfn), None)
    RMSE.append(round(rmse, 3))

# Plotting the reconstruction errors in terms of RMSE considering the different values of l (=1, 2, ..., 8)
components = np.linspace(1, 8, 8)
plt.plot(components, RMSE)
plt.title('Reconstruction error in RMSE vs l')
plt.xlabel('number of components')
plt.ylabel('RMSE')
plt.show()

# (d)
# Comparing covariance matrices of original data and reconstructed data for l =8
print("Q 3 (d):")
cov_original = pd.DataFrame(df.cov().T)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print("The covariance matrix of original data is\n", cov_original.round(3))
# The covariance matrix of reconstructed data is already computed in (c) part