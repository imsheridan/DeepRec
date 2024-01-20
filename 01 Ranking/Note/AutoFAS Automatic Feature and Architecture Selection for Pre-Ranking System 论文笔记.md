# AutoFAS: Automatic Feature and Architecture Selection for Pre-Ranking System 论文笔记

[论文](https://arxiv.org/abs/2205.09394)

## Motivation

##### 先前的方法没有明确地对性能收益与计算开销进行建模，在预排序阶段的延迟约束会导致次优的结果；从教师模型中迁移知识到预先定义好结构的学生模型中也会对模型的结果造成影响。

##### AutoFAS第一次同时选择最有价值的特征和使用神经结构搜索选择网络结构，其能够在排序教师模型的帮助下选择最合适的预排序结构。

![image-20230213201512193](C:\Users\QinHsiu\AppData\Roaming\Typora\typora-user-images\image-20230213201512193.png)

## Model

![image-20230213212911390](C:\Users\QinHsiu\AppData\Roaming\Typora\typora-user-images\image-20230213212911390.png)

##### 其模型结构主要包含两部分，左边的排序教师模型和右边的预排序学生模型，左边的教师模型主要用于特征的选择以及表征的学习，预排序网络结构中主要是多种不同的多层感知机，感知机的输入结构不一样，通过结构搜索技术为该学生模型选择合适的架构，避免了人工搜索带来的问题。其算法流程如下所示：输入相关的数据，先训练一个排序模型，然后使用训练好的排序模型作为教师模型，然后训练网络更新Mask和L参数，在每一个层中选择最重要的特征和结构，然后使用知识蒸馏训练该选中的结构，输出预先排序模型

![image-20230213214002972](C:\Users\QinHsiu\AppData\Roaming\Typora\typora-user-images\image-20230213214002972.png)

## Performance

##### 从实验结果可以看出使用该方法不仅提高了精度，还大大减少了内存消耗和时间延迟，从表4可以看出其提升了精度，并且其资源消耗以及时间延迟较与baseline都是可以比较的

![image-20230213214043421](C:\Users\QinHsiu\AppData\Roaming\Typora\typora-user-images\image-20230213214043421.png)

![image-20230213214156777](C:\Users\QinHsiu\AppData\Roaming\Typora\typora-user-images\image-20230213214156777.png)

## Ablation Study

##### 从实验结果看出其每一个板块都是有用的，并且在增大学生网络参数的时候其结构化搜索会使得模型消耗更少的时间

![image-20230213214216508](C:\Users\QinHsiu\AppData\Roaming\Typora\typora-user-images\image-20230213214216508.png)

## Conclusion

##### 这篇工作提出了一种端到端的自动化机器学习预排序模型。与简单地考虑特征的连接不同的是，该模型同时地选择特征和模型结构，联合优化使得其在计算开销和表现上都取得不错的效果，另外使用知识蒸馏技术从教师模型中学习有用的知识用于预排序。

## References

