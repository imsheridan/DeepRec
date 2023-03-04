# 逻辑回归做推荐预测任务

### 1.逻辑回归的含义

##### 逻辑回归是线性回归的衍生，线性回归的输出为预测值，逻辑回归的输出为转换后的概率值，一般使用sigmod函数求解，其中逻辑logit是log odds对数几率的意思，几率表示的是正样本的相对可能性；

### 2.逻辑回归的表达式以及求导

### $$线性回归的表达式:y=\omega x+b$$

### $$y=\sigma(\omega^{T} x+b)=\frac{1}{1+\exp(-\omega^{T} x-b)},y表示模型输出的预测概率,\ln\frac{y}{1-y}=\omega^{T}x+b,\frac{y}{1-y}表示几率，其含义是预测x为正例的概率的相对可能性,\ln(\frac{y}{1-y})表示对数几率，也即logit$$

##### 给定一个二分类任务y={0,1}，假设就y的后验概率估计分别为p(y=1|x)和p(y=0|x)，则对应的表达式可以改写为：

### $$\ln\frac{p(y=1|x)}{p(y=0|x)}=\omega^{T}x+b,p(y=1|x)=\frac{\exp(\omega^{T}x+b)}{1+\exp(\omega^{T}x+b)},p(y=0|x)=\frac{1}{1+\exp(\omega^{T}x+b)}$$

##### 优化目标是为了让模型尽可能预测出正确的类别，可以通过极大似然来估计对应的omega和bias，求解步骤如下所示：

### $$假设\beta=(\omega;b),\hat{x}=(x;1),\beta^{T}\hat{x}=\omega^{T}x+b,再令p_{1}(y=1|x)=p_{1}(y=1|\hat{x};\beta)=\frac{\exp(\beta^{T}\hat{x})}{1+\beta^{T}\hat{x}},p_{0}(y=0|x)=1-p_{1}(y=1|\hat{x};\beta)=\frac{1}{1+\exp(\beta^{T}\hat{x})}$$

### $$优化目标为最大化对数似然,\mathcal{l}(\omega,b)=\sum\limits^{m}_{i=1}\ln p(y_{i}|\hat{x};\beta)$$

### $$极大似然可以重写为:p(y_{i}|x_{i};\beta)=y_{i}p_{1}(\hat{x};\beta)+(1-y_{i})p_{0}(\hat{x};\beta))$$

### $$综合上面两个式子,最大化对数似然相当于最小化\mathcal{l}(\beta)=\sum^{m}\limits_{i=1}(-y_{i}\beta^{T}\hat{x}+\ln(1+\exp(\beta^{T}\hat{x})))$$

### $$其中\beta为高阶可导连续凸函数,由凸优化理论可知^{2},其最优解可以由梯度下降算法和牛顿法获得,\beta^{\star}=\arg\max\limits_{\beta}(\mathcal{l}(\beta)),其牛顿法迭代公式如下所示:$$

### $$牛顿法:\beta^{t+1}=\beta^{t}-(\frac{\partial^{2}\mathcal{l(\beta)}}{\partial \beta \partial \beta^{T}})^{-1}\frac{\partial \mathcal{l}(\beta)}{\partial \beta},一阶导数:\frac{\partial\mathcal{l(\beta)}}{\partial \beta}=-\sum^{m}\limits_{i=1}\hat{x_{i}}(y_{i}-p_{1}(\hat{x_{i}};\beta)),二阶导数:\frac{\partial^{2}\mathcal{l(\beta)}}{\partial \beta \partial \beta^{T}}=\sum^{m}\limits_{i=1}\hat{x_{i}}\hat{x}_{i}^{T}p_{1}(\hat{x_{i}};\beta)(1-p_{1}(\hat{x_{i}};\beta))$$

### $$梯度下降法:\beta^{t+1}=\beta^{t}-\gamma\frac{\partial \mathcal{l}(\beta)}{\partial\beta}$$

### 3.逻辑回归的一般步骤

- 构建和处理训练数据，将所有特征转换为数值型特征向量
- 确立逻辑回归模型的优化目标，建立逻辑回归模型（确定是二分类还是多分类模型）
- 模型训练，优化和更新模型参数
- 对测试集做出相关预测，并根据预测结果对商品进行排序，返回推荐列表

### 4.逻辑回归的优缺点

- 优点
  - 有数学支撑，其假设因变量y服从伯努利分布（0-1分布，n重二项式分布），线性模型假设y服从高斯分布，明显不适用于分类问题，可解释性强，其为每一个特征分配不同的权重，考虑了预测过程中不同特征的重要性不同
  - 实现简单，分类时计算量小，速度快，存储资源消耗低，易于理解和实现，广泛应用于工业问题
  - 便于观察样本分类概率分数
  - 可以结合L2正则化技术来缓解多重共线性问题

- 缺点
  - 特征空间较大的时候其性能会有所下降，难以处理大量多类特征或变量
  - 表达能力差，无法进行特征交叉、特征筛选等一系列更具可解释性的操作，因而造成信息的损失，容易欠拟合，其准确度一般不高
  - 大多处于二分类问题，使用softmax可以处理多分类，但主要用于线性可分的情况
  - 对于非线性特征，需要进行特征转换

### 5.LR的演化版本

- GBDT+LR
  - GBDT是由多棵回归树组成的树林，后一棵树以前面树林的结果与目标结果的残差作为拟合目标，例如当前有三棵树，其拟合结果为T3(x)=t1(x)+t2(x)+t3(x)，目标值为F(x)，则当前的残差为R(x)=F(x)-T3(x)，第四棵树的拟合目标即为R(x)，其每一棵树的生成过程是一棵树的标准的回归树生成过程，因此其回归树中每一个节点的分裂是一个自然的特征选择的过程，而多层节点的结构则对特征进行了有效的自动组合，高效地简化了特征选择和特征组合所带来的繁琐问题；
  - 该方法分两步进行，第一步使用GBDT进行特征组合，第二步使用LR进行预估；
  - 优点
    - 特征工程模型化，减少了复杂的人力劳动（对数据的处理，对模型中特征交叉的设计等）；
    - 决策树的深度决定了特征交叉的阶数，使用三层就可以完成特征的三阶交叉，其缓解了FM不能进行高阶特征交叉的问题，大大提升了模型的泛化性能；
  - 缺点
    - GBDT容易发生过拟合，另外使用该方法会丢失大量特征的数值信息；
    - 无法完全进行并行训练，更新参数所需的训练时间较长；

- LS-PLM;MLR

  - 其在逻辑回归的基础上引入分治的思想，其分两步骤进行，第一步先对全量数据进行聚类，第二步对每一个类别单独进行逻辑回归预测；

  - ### $$y=\sum^{m}\limits_{i=1}\pi_{i}(x)\cdot\eta_{i}(x)=\sum^{m}\limits_{i=1}softmax(x)\sigma(x)=\sum^{m}\limits_{i=1}\frac{\exp(\mu_{i}\cdot x)}{\sum^{m}\limits_{j=1}\exp(\mu_{j}\cdot x)}\cdot\frac{1}{1+\exp(-\omega_{i} x)}$$

  - 其中m表示聚类的数目，m越大其分的粒度越细，可以较好地平衡模型的拟合和推广能力，m=1时其退化为简单的逻辑回归模型；另外其模型参数规模也随m的增大而线性增长，模型收敛所需的训练样本也随之增长；

  - 优点

    - 端到端的非线性的拟合能力：其具有样本分片的能力，可以挖掘数据中的非线性模式，省去了大量的人工处理样本和特征工程的过程，使得该算法可以端到端的进行训练，便于用一个全局模型对不同应用领域、业务场景进行统一建模；
    - 模型的稀疏性强：其在建模过程中引入了L1和L2范数，可以使得最终训练出来的模型具有较高的稀疏度，使得模型的部署更加轻量级；模型服务过程中仅需使用权重非0的特征，因此稀疏模型也使其在线推断效率更高；
    - 模型架构类似于三层神经网络，具备较强的表达能力；

  - 缺点

    - 模型结构相对简单，有进一步提高的空间

### 6.python实现

- 使用sklearn自带的包，使用案例：[LogisticRegression](https://github.com/QinHsiu/DataScience_Basic/tree/main/02 Sklearn)

  ```python
  from sklearn.linear_model import LogisticRegression
  ```

- 使用python实现，代码链接：[Logistic_Regression](https://github.com/QinHsiu/DataScience_Basic/tree/main/03 Maching_Learning_in_Action/chapter 05-Logistic_Regression)

### 7.参考资料

[1] 周志华《机器学习》

[2] [Algorithms for Convex Optimization](https://convex-optimization.github.io/)

[3] 王喆《深度学习推荐系统》





