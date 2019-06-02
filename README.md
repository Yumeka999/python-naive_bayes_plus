# <center>  这是什么程序？What is this ? </center>


不同于scikit-learn的NaiveBayes输入只能是离散型变量或连续性变量, naive_bayes_plus输入既可以有离散型变量也可以有连续性变量的

Different with NaiveBayes of scikit-learn which just accept discrete attributes or continuous attributes as input, the naive_bayes_plus could accept discrete attributes and continuous attributes as input in same time.


# 1 *中文版(简体)*
## 1.1 *运行环境*
Python3

## 1.2 *Python依赖包*
|包名| 功能 | Pip 安装命令|
|:----:|----|----|
|numpy|最流行的数值计算包|Pip install numpy|


## 1.3 *如何运行?*

下面是一个训练集


|性别|身高(英尺)|体重(磅)|脚掌(英寸)|喜欢玩游戏？|
|:---:|:---:|:---:|:---:|:---:|
|男|6|180|12|否|
|男|5.92|190|11|是|
|男|5.58|170|12|是|
|男|5.92|165|10|是|
|女|5|100|6|否|
|女|5.5|150|8|否|
|女|5.42|130|7|否|
|女|5.75|150|9|是|

请用朴素贝叶斯算法预测下面这个人是男还是女？
* 身高=6
* 体重=130
* 脚掌=8
* 喜欢玩游戏？=是

**输入的Python代码**
```
from naive_bayes_plus import NavieBayesPlus

nbp = NavieBayesPlus()
l_train_x_dat = [] 
l_train_x_dat.append([6, 180, 12, 'False'])
l_train_x_dat.append([5.92, 190, 11, 'True'])
l_train_x_dat.append([5.58, 170, 12, 'True'])
l_train_x_dat.append([5.92, 165, 10, 'Trueb'])
l_train_x_dat.append([5, 100, 16, 'False'])
l_train_x_dat.append([5.5, 150, 8, 'False'])
l_train_x_dat.append([5.42, 130, 7, 'False'])
l_train_x_dat.append([5.75, 150, 9, 'True'])

l_train_y_dat = ['Male','Male','Male','Male','Female','Female','Female','Female']

nbp.train(l_train_x_dat, l_train_y_dat)
l_y, l_y_prob = nbp.predict([[6, 130, 8, 'True']])
print(l_y)
print(l_y_prob)
```



**程序输出为**
```
['Female']
[{'Male': 9.917348375732059e-11, 'Female': 1.1457231805275285e-07}]
```

**更多细节请看_test_code.py**


# 2 *Run Enviroment*
## 2.1 *Run Enviroment*
Python3

## 1.2 *Python package*
|name| function | Pip command|
|:----:|----|----|
|numpy|most popular numercial calculation package|Pip install numpy|


## 1.3 *How to run?*

This is a train set


|Gender|Hight |Weight|Footer|like playing game?|
|:---:|:---:|:---:|:---:|:---:|
|Male|6|180|12|False|
|Male|5.92|190|11|True|
|Male|5.58|170|12|True|
|Male|5.92|165|10|True|
|Female|5|100|6|False|
|Female|5.5|150|8|False|
|Female|5.42|130|7|False|
|Female|5.75|150|9|True|

So please use Naive-Bayes to predict the person is male or female.
* Hight=6
* Weight=130
* Footer=8
* like playing game?=True


**Input Python Code**
```
from naive_bayes_plus import NavieBayesPlus

nbp = NavieBayesPlus()
l_train_x_dat = [] 
l_train_x_dat.append([6, 180, 12, 'False'])
l_train_x_dat.append([5.92, 190, 11, 'True'])
l_train_x_dat.append([5.58, 170, 12, 'True'])
l_train_x_dat.append([5.92, 165, 10, 'Trueb'])
l_train_x_dat.append([5, 100, 16, 'False'])
l_train_x_dat.append([5.5, 150, 8, 'False'])
l_train_x_dat.append([5.42, 130, 7, 'False'])
l_train_x_dat.append([5.75, 150, 9, 'True'])

l_train_y_dat = ['Male','Male','Male','Male','Female','Female','Female','Female']

nbp.train(l_train_x_dat, l_train_y_dat)
l_y, l_y_prob = nbp.predict([[6, 130, 8, 'True']])
print(l_y)
print(l_y_prob)
```



**Output**
```
['Female']
[{'Male': 9.917348375732059e-11, 'Female': 1.1457231805275285e-07}]
```

**more details in _test_code.py**
