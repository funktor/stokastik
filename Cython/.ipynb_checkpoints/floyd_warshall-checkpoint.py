def floydWarshall(dist_mat, n):
    for k in range(n): 
        for i in range(n):
            for j in range(n):
                dist_mat[i][j] = min(dist_mat[i][j], dist_mat[i][k] + dist_mat[k][j]) 

    return dist_mat