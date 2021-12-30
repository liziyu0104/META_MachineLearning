# WoRks ```continued``` 

```English``` ``` 中文``` ```Français```

# Probability Density Estimation-Non-Parametric Methods(概率密度估计-非参数方法)
## 1. Kernel / k-Nearest Neighborhood Density Estimators (核密度估计 /  K邻近密度估计)
* KDE:  Fix volume, determine number of points in this volume
* K-NN: Fix the number of points and increase the volume to include this number of points
```python
python apply.py
```
![R](https://raw.githubusercontent.com/liziyu0104/META_MachineLearning/main/Probability_Density_Estimation/result/R1.png)

## 2. Expectation Maximization (EM) Algo for Gaussian Mixture Model (GMM) （应用于高斯混合模型的期望最大化算法）
* EM algo is sensible to init, we can use k-means fist for some steps ```kmeans = KMeans(n_clusters = K, n_init = 10).fit(data)```
```python
python apply.py
```
![R](https://raw.githubusercontent.com/liziyu0104/META_MachineLearning/main/Probability_Density_Estimation/result/R2.jpg)
![R](https://raw.githubusercontent.com/liziyu0104/META_MachineLearning/main/Probability_Density_Estimation/result/R3.png)
