import numpy as np
from scipy import special

# S:重叠积分矩阵
S = np.zeros([2, 2])
# C:基组的系数矩阵
C = np.zeros([3, 2])
# A:基组的轨道系数矩阵
A = np.zeros([3, 2])
# R:分子原子的笛卡尔坐标矩阵
R = np.zeros([3, 2])
# T:分子的动能积分矩阵
T = np.zeros([2, 2])
# V：分子的核吸引积分矩阵
V = np.zeros([2, 2])
# Z：分子的电荷
Z = np.zeros([2])
# 高斯函数收缩系数
C[0, 0] = 0.444635
C[1, 0] = 0.535328
C[2, 0] = 0.154329
C[0, 1] = 0.444635
C[1, 1] = 0.535328
C[2, 1] = 0.154329
# 高斯函数轨道系数
A[0, 0] = 0.168856
A[1, 0] = 0.623913
A[2, 0] = 3.42525
A[0, 1] = 0.168856
A[1, 1] = 0.623913
A[2, 1] = 3.42525
# 分子的笛卡尔坐标
R[0, 0] = 0
R[1, 0] = 0
R[2, 0] = 1.4
R[0, 1] = 0
R[1, 1] = 0
R[2, 1] = 0
# 原子的电荷矩阵
Z[0] = 1
Z[1] = 1


# 重叠积分
def Soverlap():
    i = 0
    j = 0
    m = 0
    k = 0
    for i in range(2):
        for j in range(2):
            for m in range(3):
                for k in range(3):
                    S[i, j] = S[i, j] + C[m, i] * C[k, j] * (
                            4 * A[m, i] * A[k, j] / ((A[m, i] + A[k, j]) ** 2)
                    ) ** 0.75 * (
                                  np.exp(-A[m, i] * A[k, j] * (
                                          (R[0][i] - R[0, j]) ** 2 +
                                          (R[1][i] - R[1, j]) ** 2 +
                                          (R[2][i] - R[2, j]) ** 2) / (
                                                 A[m, i] + A[k, j]))
                              )
    return S


# 动能矩阵
def Tmatrix():
    i = 0
    j = 0
    m = 0
    k = 0
    for i in range(2):
        for j in range(2):
            for m in range(3):
                for k in range(3):
                    T[i][j] = T[i][j] + C[m][i] * C[k][j] * (
                            (4 * A[m][i] * A[k][j] / ((A[m][i] + A[k][j]) ** 2)) ** 0.75) * (
                                      np.exp(-A[m][i] * A[k][j] * (
                                              (R[0][i] - R[0, j]) ** 2 +
                                              (R[1][i] - R[1, j]) ** 2 +
                                              (R[2][i] - R[2, j]) ** 2) / (A[m][i] + A[k][j])) * A[k][j] * (
                                              3 * A[m][i] / (A[m][i] + A[k][j]) - 2 * A[k][j] * (
                                          (((A[m][i] * R[0][i] + A[k][j] * R[0][j]) / (A[m][i] + A[k][j]) - R[0][
                                              j]) ** 2 +
                                           ((A[m][i] * R[1][i] + A[k][j] * R[1][j]) / (A[m][i] + A[k][j]) - R[1][
                                               j]) ** 2 +
                                           ((A[m][i] * R[2][i] + A[k][j] * R[2][j]) / (A[m][i] + A[k][j]) - R[2][
                                               j]) ** 2)
                                      )
                                      )
                              )
    return T


# 核吸引矩阵
def Vmatrix():
    i = 0
    j = 0
    m = 0
    k = 0
    c = 0
    for i in range(2):
        for j in range(2):
            for m in range(3):
                for k in range(3):
                    for c in range(2):
                        if (((A[m][i] * R[0][i] + A[k][j] * R[0][j]) / (A[m][i] + A[k][j]) - R[0][c]) ** 2
                            + ((A[m][i] * R[1][i] + A[k][j] * R[1][j]) / (A[m][i] + A[k][j]) - R[1][c]) ** 2
                            + ((A[m][i] * R[2][i] + A[k][j] * R[2][j]) / (A[m][i] + A[k][j]) - R[2][c]) ** 2
                        ) ** 0.5 == 0:
                            V[i][j] = V[i][j] + C[m][i] * C[k][j] * (-2) * Z[c] * np.exp(-A[m][i] * A[k][j] * (
                                    (R[0][i] - R[0, j]) ** 2 +
                                    (R[1][i] - R[1, j]) ** 2 +
                                    (R[2][i] - R[2, j]) ** 2) / (A[m][i] + A[k][j])) / (
                                              (A[m][i] + A[k][j]) * (np.pi) ** 0.5) * (
                                              (4 * A[m][i] * A[k][j]) ** 0.75
                                      )
                        else:
                            V[i][j] = V[i][j] + C[m][i] * C[k][j] * (
                                -(4 * A[m][i] * A[k][j] / ((A[m][i] + A[k][j]) ** 2)) ** 0.75) * Z[c] * (
                                              np.exp(-A[m][i] * A[k][j] * (
                                                      (R[0][i] - R[0, j]) ** 2 +
                                                      (R[1][i] - R[1, j]) ** 2 +
                                                      (R[2][i] - R[2, j]) ** 2) / (A[m][i] + A[k][j])) /
                                              (((A[m][i] * R[0][i] + A[k][j] * R[0][j]) / (A[m][i] + A[k][j]) - R[0][
                                                  c]) ** 2
                                               + ((A[m][i] * R[1][i] + A[k][j] * R[1][j]) / (A[m][i] + A[k][j]) - R[1][
                                                          c]) ** 2
                                               + ((A[m][i] * R[2][i] + A[k][j] * R[2][j]) / (A[m][i] + A[k][j]) - R[2][
                                                          c]) ** 2
                                               ) ** 0.5 * special.erf(
                                          (((A[m][i] * R[0][i] + A[k][j] * R[0][j]) / (A[m][i] + A[k][j]) - R[0][
                                              c]) ** 2
                                           + ((A[m][i] * R[1][i] + A[k][j] * R[1][j]) / (A[m][i] + A[k][j]) - R[1][
                                                      c]) ** 2
                                           + ((A[m][i] * R[2][i] + A[k][j] * R[2][j]) / (A[m][i] + A[k][j]) - R[2][
                                                      c]) ** 2
                                           ) ** 0.5 * (A[m][i] + A[k][j]) ** 0.5)

                                      )
    return V


# 哈密顿矩阵
def Hmatrix():
    return Tmatrix() + Vmatrix()


# 双电子矩阵
def Double_electron():
    D = np.zeros([2, 2, 2, 2])
    RL = np.zeros([3])
    RL_distance = 0
    a = 0
    b = 0
    c = 0
    d = 0
    r = 0
    s = 0
    t = 0
    u = 0
    for r in range(2):
        for s in range(2):
            for t in range(2):
                for u in range(2):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                for d in range(3):
                                    Rp1 = np.zeros([3])
                                    Rp2 = np.zeros([3])
                                    # 算向量Rp1的x,y,z坐标
                                    Rp1[0] = Rp1[0] + (A[a][r] * R[0][r] + A[b][s] * R[0][s]) / (A[a][r] + A[b][s])
                                    Rp1[1] = Rp1[1] + (A[a][r] * R[1][r] + A[b][s] * R[1][s]) / (A[a][r] + A[b][s])
                                    Rp1[2] = Rp1[2] + (A[a][r] * R[2][r] + A[b][s] * R[2][s]) / (A[a][r] + A[b][s])
                                    # 算向量Rp2的x,y,z坐标
                                    Rp2[0] = Rp2[0] + (A[c][t] * R[0][t] + A[d][u] * R[0][u]) / (A[c][t] + A[d][u])
                                    Rp2[1] = Rp2[1] + (A[c][t] * R[1][t] + A[d][u] * R[1][u]) / (A[c][t] + A[d][u])
                                    Rp2[2] = Rp2[2] + (A[c][t] * R[2][t] + A[d][u] * R[2][u]) / (A[c][t] + A[d][u])
                                    # C[a][0]*C[b][0]*C[c][0]*C[d][0]
                                    RL = Rp1 - Rp2
                                    RL_distance = (RL[0] ** 2 + RL[1] ** 2 + RL[2] ** 2) ** 0.5

                                    if RL_distance == 0:
                                        D[r][s][t][u] = D[r][s][t][u] + C[a][r] * C[b][s] * C[c][t] * C[d][u] * 64 * (
                                            np.exp(-A[a][r] * A[b][s] * (
                                                    0 + (R[0][r] - R[0][s]) ** 2 + (R[1][r] - R[1][s]) ** 2 + (
                                                    R[2][r] - R[2][s]) ** 2) / (A[a][r] + A[b][s]))) * (
                                                            np.exp(
                                                                -A[c][t] * A[d][u] * (0 + (R[0][t] - R[0][u]) ** 2 + (
                                                                        R[1][t] - R[1][u]) ** 2 + (R[2][t] - R[2][
                                                                    u]) ** 2) / (A[c][0] + A[d][0]))) * (
                                                                (A[a][r] * A[b][s] * A[c][t] * A[d][u]) ** 0.75 * (
                                                                1 / (4 * (A[a][r] + A[b][s]) * (
                                                                A[c][t] + A[d][u]))) ** 1.5 * (np.pi * (
                                                                (A[a][r] + A[b][s] + A[c][t] + A[d][u]) / (
                                                                4 * (A[a][r] + A[b][s]) * (A[c][t] + A[d][u])))) ** (
                                                                    -0.5)
                                                        )
                                    else:
                                        D[r][s][t][u] = D[r][s][t][u] + C[a][r] * C[b][s] * C[c][t] * C[d][u] * 64 * (
                                            np.exp(-A[a][r] * A[b][s] * (
                                                    0 + (R[0][r] - R[0][s]) ** 2 + (R[1][r] - R[1][s]) ** 2 + (
                                                    R[2][r] - R[2][s]) ** 2) / (A[a][r] + A[b][s]))) * (
                                                            np.exp(
                                                                -A[c][t] * A[d][u] * (0 + (R[0][t] - R[0][u]) ** 2 + (
                                                                        R[1][t] - R[1][u]) ** 2 + (R[2][t] - R[2][
                                                                    u]) ** 2) / (A[c][t] + A[d][u]))) * (
                                                                (A[a][r] * A[b][s] * A[c][t] * A[d][u]) ** 0.75 * (
                                                                1 / (4 * (A[a][r] + A[b][s]) * (
                                                                A[c][t] + A[d][u]))) ** 1.5 * special.erf(
                                                            RL_distance / (
                                                                    2 * ((A[a][r] + A[b][s] + A[c][t] + A[d][u]) / (
                                                                    4 * (A[a][r] + A[b][s]) * (
                                                                    A[c][t] + A[d][u]))) ** 0.5))) / RL_distance
    return D


# 重叠积分矩阵的+0.5次方
def S_power_positive_half(S1):
    eigen_value, eigen_vector = np.linalg.eig(S1)
    eigen_value = np.linalg.eigvalsh(S1)
    eigen_value_hf = np.diag(eigen_value ** (0.5))
    S_positive_half = np.matmul(np.linalg.inv(eigen_vector), eigen_value_hf)
    S_positive_half = np.matmul(S_positive_half, eigen_vector)
    return S_positive_half


# 重叠积分矩阵的-0.5次方
def S_power_negtive_half(S1):
    #这一块数学查过来查过去，表明基础很弱鸡
    eigen_value, eigen_vector = np.linalg.eig(S1)
    eigen_value = np.linalg.eigvalsh(S1)
    eigen_value_hf = np.diag(eigen_value ** (-0.5))
    S_negtive_half = np.matmul(np.linalg.inv(eigen_vector), eigen_value_hf)
    S_negtive_half = np.matmul(S_negtive_half, eigen_vector)
    return S_negtive_half


# 构建密度矩阵
def denstyMatrix(guess):
    P1 = np.zeros([2, 2])
    t = 0
    u = 0
    j = 0
    for t in range(2):
        for u in range(2):
            for j in range(1):
                P1[t][u] = P1[t][u] +  2*guess[t][j] * guess[u][j]
    return P1


S = Soverlap()
S_negtive_hf = S_power_negtive_half(S)
S_positive_hf = S_power_positive_half(S)
H1 = Hmatrix()
D = Double_electron()
guess = np.zeros([2, 2])
guess[0][0] = 0.548934
guess[0][1] = -1.211463
guess[1][0] = 0.548934
guess[1][1] = 1.211463


# 构建Fock矩阵

def F(guess):
    F = np.zeros([2, 2])

    # 啊这一块简直了，这特么H1是个对象，不能简单地赋值，debug发现H1矩阵居然会发生变化，
    # 奇了怪了，然后才用下面这个循环来赋值
    i = 0
    j = 0
    for i in range(2):
        for j in range(2):
            F[i][j] = H1[i][j]
    P1 = denstyMatrix(guess)

    r = 0
    s = 0
    t = 0
    u = 0
    for r in range(2):
        for s in range(2):
            for t in range(2):
                for u in range(2):
                    F[r][s] = F[r][s] + P1[t][u] * (D[r][s][t][u] - 0.5*D[r][u][t][s])

    # 构建 F‘矩阵
    F1 = np.matmul(S_negtive_hf, F)
    F1 = np.matmul(F1, S_negtive_hf)
    eigvalue, eigvactor = np.linalg.eig(F1)
    guess1 = np.matmul(S_negtive_hf, eigvactor)
    print(eigvalue)
    print("----------------------------------------------------------")
    return eigvalue, guess1


i = 0
for i in range(100):
    eigvalue, guess = F(guess)
