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



## 二、实验原理

### 1.强化学习

​		强化学习（Reinforcement Learning，RL），也叫增强学习，是指一类从（与环境）交互中不断学习的问题以及解决这类问题的方法．强化学习问题可以描述为一个智能体从与环境的交互中不断学习以完成特定目标。和深度学习类似，强化学习中的关键问题也是贡献度分配问题 [Minsky, 1961]，每一个动作并不能直接得到监督信息，需要通过整个模型的最终监督信息（奖励）得到，并且有一定的延时性。

​		在强化学习中，有两个可以进行交互的对象：智能体和环境． 

+ 智能体（Agent）可以感知外界环境的状态（State）和反馈的奖励 (Reward），并进行学习和决策．智能体的决策功能是指根据外界环境的状态来做出不同的动作（Action），而学习功能是指根据外界环境的奖励来调整策略． 

+ 环境（Environment）是智能体外部的所有事物，并受智能体动作的影响而改变其状态，并反馈给智能体相应的奖励．

![1670751040846](C:\Users\Yan xinyi\AppData\Roaming\Typora\typora-user-images\1670751040846.png)

### 2.Deep Q-Networks

​		Q学习（Q-Learning）算法[Watkins et al., 1992]是一种异策略的时序差分学习方法。在Q学习中，Q函数的估计方法为
$$
Q\left( s,a \right) ←Q\left( s,a \right) +\alpha \left( r+\gamma \underset{a'}{\max}Q\left( s',a' \right) -Q\left( s,a \right) \right)
$$
相当于让Q(s,a)直接去估计最优状态值函数Q(s,a)。

​		依赖Q学习的目标函数存在目标不稳定，参数学习的目标依赖于参数本身以及样本之间有很强的相关性的两个问题。为了解决这两个问题，[Mnih et al., 2015] 提出了一种深度 Q 网络（Deep Q-Networks，DQN）．深度 Q 网络采取两个措施：

+ 目标网络冻结（Freezing Target Networks），即在一个时间段内固定目标中的参数，来稳定学习目标；经验回放可以形象地理解为在回忆中学习． 

+ 经验回放（Experience Replay），即构建一个经验池（Replay Buffer）来去除数据相关性．经验池是由智能体最近的经历组成的数据集． 

​        训练时，随机从经验池中抽取样本来代替当前的样本用来进行训练．这样，就打破了和相邻训练样本的相似性，避免模型陷入局部最优．经验回放在一定程度上类似于监督学习．先收集样本，然后在这些样本上进行训练。

### 3.深度强化学习

​		在计算机视觉与语音识别方面的突破依赖于非常大的训练集上有效地训练深度神经网络。最成功的方法是**直接从原始输人进行训练，使用基于随机梯度下降的轻型更新**。通过向深度神经网络输人足够的数据，通常可以学到比手工制作的特征更好的表征。这些成功的经验促使我们采取强化学习的方法。我们的目标是**将强化学习算法连接到一个深度神经网络，该网络直接在RGB图像上操作，并通过使用随机梯度更新来有效地处理训练数据**。

​		Tesauro的TD-Gammon架构为这种方法提供了一个出发点。这个架构更新了一个估计价值函数的网络的参数，这些参数直接来自政策上的经验样本，St，at，rt，St+1,at+1，这些经验来自于算法与环境的互动。由于这种方法在20年前就能超过最好的人类双陆棋手，我们自然会想，20年的硬件改进，加上现代深度神经网络架构和可扩展的RL算法，是否会产生重大进展。
​		与TD-Gammon和类似的在线方法相比，我们利用一种被称为经验重放的技术(experience replay)，我们将智能体在每个时间步的经验，et = (St ，at ，rt ，St+1 )存储在一个数据集D = e1 ,…，e2中，将许多情节汇聚到一个重放储存器中。

​		在算法的循环中，我们对经验样本应用Q-learning更新，或称minibatch更新，从存储的样本池中随机抽取。在进行经验回放后，代理人根据E-贪婪政策选择并执行行动。由于使用历史上的补丁长度作为神经网络的输人，任意长度的输人可能是困难的，我们的Q-函数反而在由函数ɸ 产生的历史的固定长度表示上工作。 完整的算法，我们称之为深度Q-learning。这种方法比标准的在线Q-learning有几个优点：

+ 每一步经验都有许多权重更新，这使得数据效率更高
+ 随机化的样本打破了连续样本的关联性，减少了更新方差
+ 在参数学习时，当前的参数决定了参数训练的下一个数据样本

![1670768885908](C:\Users\Yan xinyi\AppData\Roaming\Typora\typora-user-images\1670768885908.png)

​		在无监督的情况下，agent不断从一个状态转至另一状态进行探索，直到到达目标。我们将agent的每一次探索（从任意初始状态开始，经历若干action，直到到达目标状态的过程）称为一个**episode**。

​		不难看出，不需要的反馈回路可能会出现，参数可能陷入局部最小值，甚至出现灾难性的发散。通过使用经验重放，行为分布在其以前的许多状态中被平均化，使学习变得平滑，避免了参数的振荡或发散。当通过经验回放学习时，有必要进行非政策性学习(因为我们当前的参数与用于生成样本的参数不同)，这就是选择Q-learning的动机。

### 4.DDQN

​		DDQN是Double DQN的缩写，是在DQN的基础上改进而来的。DDQN的模型结构基本和DQN的模型结构一模一样，唯一不同的就是它们的目标函数。DoubleDQN的最优动作选择是根据当前正在更新的Q网络的参数θ t ,而DQN中的最优动作选择是根据上一小节提到的target-Q网络的参数θ t − \theta_t^-θ 。

​		由于传统的DQN通常会高估Q值的大小（overestimation），而DDQN由于每次选择的根据是当前Q网络的参数，并不是像DQN那样根据target-Q的参数，所以当计算target值时是会比原来小一点的。（因为计算target值时要通过target-Q网络，在DQN中原本是根据target-Q的参数选择其中Q值最大的action，而现在用DDQN更换了选择以后计算出的Q值一定是小于或等于原来的Q值的）这样在一定程度上降低了overestimation，使得Q值更加接近真实值。

![1670939646625](C:\Users\Yan xinyi\AppData\Roaming\Typora\typora-user-images\1670939646625.png)

### 5.Dueling DQN

​		Dueling DQN也是在DQN的基础上进行的改进。改动的地方是DQN神经网络的最后一层，原本的最后一层是一个全连接层，经过该层后输出n个Q值（n代表可选择的动作个数）。而Dueling DQN不直接训练得到这n个Q值，它通过训练得到的是两个间接的变量V(state value)和A(action advantage)，然后通过它们的和来表示Q值。

​		研究动机是在Q-learning中，对于很多状态，没有必要估计每个动作选择的价值：在某些状态下，知道采取哪种行动很重要；但是在很多其他状态，行动的选择对所要发生的事情可能没有影响。然而，对于基于bootstrapping的Q-learning算法，状态值的估计对于每个状态都非常重要。



## 三、实验步骤

1.获取项目代码

2.搭建实验环境

​	使用Anaconda3安装一个python==3.7的虚拟Python版本（在Anaconda3 Prompt中进行）

​	 `conda create -n tf_new python=3.7`

​	激活环境后安装对应版本的tensorflow

​	`conda install tensorflow==1.15.0`

​	随后在Pycharm配置对应的tensorflow环境，并使用如下指令安装gym,tqdm以及opencv2等python库

​	`Python -m pip install packcages  `

3.运行并调试代码



## 四、实验结果

### 1.DQN算法运行结果

​		DQN（Deep Q Network）是一种常用的深度强化学习算法，它通过使用神经网络来学习一个价值函数来决定在给定状态下的最优动作。breakout是一种电子游戏，玩家需要使用一个挡板来阻止一个小球掉落，同时还要击破障碍物来获得分数。DQN算法可以用来训练一个模型来玩breakout游戏，通过不断学习来提高游戏的表现。例如，在训练过程中，DQN模型可以通过尝试不同的动作来学习如何更好地控制挡板，从而获得更高的分数。在本实验中，我们选择使用Breakout来验证DQN的算法效果，运行过程图如下：

![训练过程](C:\Users\Yan xinyi\AppData\Roaming\Typora\typora-user-images\1670725019144.png)

模型训练完成后，使用excel清洗训练数据，Matlab绘图后结果如下：

​		

![avg.reward](C:\Users\Yan xinyi\Desktop\avg.reward.png)

​		在监督学习中，人们可以通过在训练集和验证集上对模型进行评估来轻松跟踪其训练过程中的表现。然而，在强化学习中，我们的评估指标是智能体在一个情节或游戏中收集的总奖励，在一些游戏中的平均值，我们在训练期间定期计算它，即average reward。average reward 指的是平均收益，即算法在训练过程中获得的总收益与训练轮数的比值。这个值可以用来衡量算法的效果，其越高表明算法的性能越好。由于微小的权重变化会导致状态巨变，average reward/epoisode图像显现出了极大的噪声。

​		由图像易知，average reward在1.7million前持续上升，在1.7million附近时达到峰值并开始趋于稳定，在后续随episode的持续训练，average reward小范围内波动。表明在1.7million时算法已经起到了很好的训练效果，并开始收敛。

![avg.loss](C:\Users\Yan xinyi\Desktop\avg.loss.png)

​		在DQN中，average loss指的是平均损失，即算法在训练过程中总的损失值与训练轮数的比值。损失值反映了算法预测的价值函数与真实价值函数之间的差距，因此average loss越小表明算法预测的价值函数越准确。

​		由图像易知，average loss在1.7million前持续上升，并在1.7million处达到峰值并开始下降，最终在训练3.2million次后稳定在0.075附近，这表明该模型已经达到了稳定的状态，即它在后续的训练中不会有太大的变化，已经达到了一个较好的状态。

![avg.q](C:\Users\Yan xinyi\Desktop\avg.q.png)

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

