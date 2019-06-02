from naive_bayes_plus import NavieBayesPlus

'''
---------------------------------------------------------
Gender      Hight   Weight  FootSize    LikePlayingGame
---------------------------------------------------------
Male        6	    180	    12          False
Male	    5.92	190	    11          True
Male  	    5.58	170	    12          True
Male	    5.92	165	    10          True
Female	    5	    100	    6           False
Female	    5.5	    150	    8           False
Female	    5.42	130	    7           False
Female	    5.75	150	    9           True
---------------------------------------------------------
Predict a unkown smaple is male or female

Hight = 6
Weight = 130
FootSize = 8
LikePlayingGame = "True"


'''


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
print(nbp.lo_x)
print(nbp.lo_y)


l_y, l_y_prob = nbp.predict([[6, 130, 8, 'True']])
print(l_y)
print(l_y_prob)