# 深度学习原理课程大作业（DQN）

## 一、系统配置

1.操作系统：Windows 11

2.编译环境：PyCharm 2020.1.1 x64      

2.python环境设置：

- python 3.7.13
- gym 0.15.7
- tqdm 4.61.2
- tensorflow 1.15.0
- protobuf 3.19.0
- opencv-python
- ale-py
- gym[accept-rom-license]
- gym[atari]

## 二、实验步骤

1.获取项目代码

2.搭建实验环境

​	使用Anaconda3安装一个python==3.7的虚拟Python版本（在Anaconda3 Prompt中进行）

​	 `conda create -n tf_new python=3.7`

​	激活环境后安装对应版本的tensorflow

​	`conda install tensorflow==1.15.0`

​	随后在Pycharm配置对应的tensorflow环境，并使用如下指令安装gym,tqdm以及opencv2等python库

​	`Python -m pip install packcages  `

3.运行并调试代码
具体报错的解决方法参考https://blog.csdn.net/qq_39698985/article/details/127922370

## 三、实验结果

### 1.DQN算法运行结果

​		DQN（Deep Q Network）是一种常用的深度强化学习算法，它通过使用神经网络来学习一个价值函数来决定在给定状态下的最优动作。breakout是一种电子游戏，玩家需要使用一个挡板来阻止一个小球掉落，同时还要击破障碍物来获得分数。DQN算法可以用来训练一个模型来玩breakout游戏，通过不断学习来提高游戏的表现。例如，在训练过程中，DQN模型可以通过尝试不同的动作来学习如何更好地控制挡板，从而获得更高的分数。在本实验中，我们选择使用Breakout来验证DQN的算法效果，运行过程图如下：

![训练过程](C:\Users\Yan xinyi\AppData\Roaming\Typora\typora-user-images\1670725019144.png)

模型训练完成后，使用excel清洗训练数据，Matlab绘图后结果如下：

![avg.reward](C:\Users\Yan xinyi\Desktop\学习资料-常用笔记\DQN\avg.reward.png)

​		在监督学习中，人们可以通过在训练集和验证集上对模型进行评估来轻松跟踪其训练过程中的表现。然而，在强化学习中，我们的评估指标是智能体在一个情节或游戏中收集的总奖励，在一些游戏中的平均值，我们在训练期间定期计算它，即average reward。average reward 指的是平均收益，即算法在训练过程中获得的总收益与训练轮数的比值。这个值可以用来衡量算法的效果，其越高表明算法的性能越好。由于微小的权重变化会导致状态巨变，average reward/epoisode图像显现出了极大的噪声。

​		由图像易知，average reward在1.7million前持续上升，在1.7million附近时达到峰值并开始趋于稳定，在后续随episode的持续训练，average reward小范围内波动。表明在1.7million时算法已经起到了很好的训练效果，并开始收敛。

![avg.loss](C:\Users\Yan xinyi\Desktop\学习资料-常用笔记\DQN\avg.loss.png)

​		在DQN中，average loss指的是平均损失，即算法在训练过程中总的损失值与训练轮数的比值。损失值反映了算法预测的价值函数与真实价值函数之间的差距，因此average loss越小表明算法预测的价值函数越准确。

​		由图像易知，average loss在1.7million前持续上升，并在1.7million处达到峰值并开始下降，最终在训练3.2million次后稳定在0.075附近，这表明该模型已经达到了稳定的状态，即它在后续的训练中不会有太大的变化，已经达到了一个较好的状态。

![avg.q](C:\Users\Yan xinyi\Desktop\学习资料-常用笔记\DQN\avg.q.png)

​		另一个更稳定的指标是参数的行动价值函数Q，它提供了智能体从任何给定状态遵循其参数可以获得多少折现奖励的什计。我们在训练开始前通过运行随机策略收集一组固定的状态，并跟踪这些状态的最大Q值的平均值，即average q。average q是算法在训练过程中预测的总价值与训练轮数的比值。在深度强化学习中，average q通常用来衡量算法的效果，其越大表明算法的性能越好。

​		由数据易知，average q在2.45million前持续增加，在2.45million附近时达到峰值但存在较大振荡，在3.2million后average q稳定在3.5附近，模型在此时达到了良好的训练效果。

​		以上三种参数的联系是：在训练过程中，average q和average reward都是通过average loss来实现改进的，即通过最小化average loss来最大化average q和average reward。

​		在训练过程中通过Breakout的分数可明显看出，随训练次数的增加，游戏积分有明显的上升，并且在模型训练结束时很好地完成了游戏通关。

<video src="D:\A_DQN\DQN展示-1.5倍速.mp4"></video>

### 2.纵向：多算法对比

​		为了进一步研究DQN算法，我们将DQN算法与DDQN、Dueling Network进行对比。运行如下代码：

```python 
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling',True, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q',True, 'Whether to use double q-learning')
```

由于训练所需时间过长，实验中并未进行完整的训练，观察训练过程可知：

​		DDQN将动作选择和策略评估分离开，降低了过高估计Q值的风险，Atari 游戏上的实验表明，DDQN可以估计出更加准确的Q值，效果比DQN更稳定。并且具有更快的收敛速度，算法的性能得到有效提升。

​		Dueling Network相比于DQN而言，将state和action进行了一定程度的分离，使state不再完全依赖于action的价值来进行判断，当环境中频繁出现采取不同action但对应值函数相等的情况时Dueling Network比DQN有更好的发挥。

​		最后，查阅论文可知，DDQN训练出的agent玩Atari游戏，从大多数游戏的表现上来看，利用DDQN训练出的效果对比DQN在得分上有显著提升，DDQN可以估计出更加准确的Q值，效果比DQN更稳定；而利用Dueling DQN训练出的agent相比于DQN也取得了更好的分数，在Atlantis ，Tennis，Space Invaders三个游戏上最好的成绩接近改进前的三倍之多。

### 3.横向：多游戏同算法

为了更全面地了解DQN算法在游戏训练上的优越性，我们继续选取多个游戏进行训练。改动如下代码：

```python
#Pong
env_name = 'Pong-v0'
flags.DEFINE_string('env_name', 'Pong-v0', 'The name of gym environment to use')
#BeamRider
env_name = 'BeamRider-v0'
flags.DEFINE_string('env_name', 'BeamRider-v0', 'The name of gym environment to use')
#SpaceInvaders
env_name = 'SpaceInvaders-v0'
flags.DEFINE_string('env_name', 'SpaceInvaders-v0', 'The name of gym environment to use')
#Qbert
env_name = 'Qbert-v0'
flags.DEFINE_string('env_name', 'Qbert-v0', 'The name of gym environment to use')
#Enduro
env_name = 'Enduro-v0'
flags.DEFINE_string('env_name', 'Enduro-v0', 'The name of gym environment to use')
```

游戏运行界面如下：![1670994531335](C:\Users\Yan xinyi\AppData\Roaming\Typora\typora-user-images\1670994531335.png)

​		在深度强化学习（DQN）中，scale是一种策略，用于对输入数据进行预处理。它的主要目的是将不同的输入特征缩放到相同的尺度，以便使用神经网络模型处理。这个过程有助于提高模型的准确度和精度。简单来说，scale将输入数据的每个特征的值映射到一个固定的范围，以便模型能够更好地处理这些数据。由于训练时间过长，我们将参数scale改为10，观察数据发现与breakout的训练效果类似。

​		最后，可以发现，DQN在Breakout、Enduro和Pong上取得了比人类专家更好的表现，但是在O*bert、Seaquest、SpaceInvaders等游戏上的表现与人类相差甚远，这些游戏更具有挑战性，因为它们需要网络找到一个延伸到长时间尺度的策略。



​		总的来说，通过本次实验，我们将强化学习算法连接到一个深度神经网络，该网络直接在RGB图像上操作，并通过使用随机梯度更新来有效地处理训练数据。深入地体会到随机小批量更新与经验重放记忆结合的算法优越性。

