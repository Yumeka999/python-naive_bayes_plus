import numpy as np

class NavieBayesPlus:
    def __init__(self):
        self.n_sample_num = 0

        self.n_x_num = 0 # total number of attribute
        self.lo_x = [] 

        self.n_y_num = 0
        self.lo_y = []      # save y(label) inof and prior probability 
        self.o_y_map = {}
        self.ln_y_index = []
           
    def __get_prior_prob_y(self, l_train_y_dat):
        o_y_info = {}
        for e in l_train_y_dat:
            if e not in o_y_info.keys():
                o_y_info[e] = 1
            else:
                o_y_info[e] += 1

        i = 0
        for k, v in o_y_info.items():
            self.lo_y.append({"y":k, "num": v, "prior_prob": v/self.n_sample_num})
            self.o_y_map[k] = i
            i += 1
        self.n_y_num = len(self.lo_y)
        self.ln_y_index = [[] for i in range(self.n_y_num)]   
        for i, y in enumerate(l_train_y_dat): 
            self.ln_y_index[self.o_y_map[y]].append(i)
        

    def __get_likelihood_prob_x(self, l_train_x_dat, l_train_y_dat):
        self.n_x_num = len(l_train_x_dat[0])
        l_x_row = l_train_x_dat[0]

        for i in range(self.n_x_num):
            o_x = {"ith": i, "type": "", "likely_info": {}}
            if isinstance(l_x_row[i], str):
                o_x["type"] = "discrete"

                l_ith_x = [row[i] for row in l_train_x_dat] # get all data of i-th col
                ls_ith_x_unique = list(set(l_ith_x)) 
                n_ith_x_attr_num = len(ls_ith_x_unique)
                
                o_x["likely_info"]["row"] = [e["y"] for e in self.lo_y]
                o_x["likely_info"]["col"] = [e for e in ls_ith_x_unique]
                o_x["likely_info"]["likely_dat_mat"] = np.zeros([self.n_y_num, n_ith_x_attr_num])
                o_x["likely_info"]["likely_prob_mat"] = np.zeros([self.n_y_num, n_ith_x_attr_num])

                for x, y in zip(l_ith_x, l_train_y_dat):
                    y_index = self.o_y_map[y]
                    x_index = ls_ith_x_unique.index(x)
                    o_x["likely_info"]["likely_dat_mat"][y_index, x_index] += 1
                



                # Get likelihood probability by Laplace smothing 
                for j in range(self.n_y_num):
                    for k in range(n_ith_x_attr_num):
                        o_x["likely_info"]["likely_prob_mat"] = (o_x["likely_info"]["likely_dat_mat"] + 1) / ( self.lo_y[j]["num"] + 2) 

                o_x["likely_info"]["likely_dat_mat"] = o_x["likely_info"]["likely_dat_mat"].tolist()
                o_x["likely_info"]["likely_prob_mat"] = o_x["likely_info"]["likely_prob_mat"].tolist()

            else:
                o_x["type"] = "continue"

                l_ith_x = [row[i] for row in l_train_x_dat] # get all data of i-th col
                
                o_x["likely_info"]["y"] = [e["y"] for e in self.lo_y]
                o_x["likely_info"]["norm_parm"] = {}

                for j, o_y in enumerate(self.lo_y):
                    o_x["likely_info"]["norm_parm"][o_y["y"]] = {}


                    o_norm_info = {"y":o_y["y"], "mean": 0.0, "sigma": 0.0, "2nd_moment": 0.0}
                    l_ith_x_with_y = [l_ith_x[k] for k in self.ln_y_index[j]]
                    np_ith_x_with_y = np.asarray(l_ith_x_with_y)

                    o_x["likely_info"]["norm_parm"][o_y["y"]]["mean"] = np.mean(np_ith_x_with_y)
                    o_x["likely_info"]["norm_parm"][o_y["y"]]["sample_var"] = ((np_ith_x_with_y - np.mean(np_ith_x_with_y)) ** 2).sum() / (np_ith_x_with_y.size - 1)
                    o_x["likely_info"]["norm_parm"][o_y["y"]]["2nd_moment"] = np.mean(np_ith_x_with_y**2)

            self.lo_x.append(o_x)
               

    def train(self, l_train_x_dat, l_train_y_dat):
        # detect the number of discrete attribute and the number of continue attribute
        self.n_sample_num = len(l_train_x_dat)

        # get prior probability 
        self.__get_prior_prob_y(l_train_y_dat)

        self.__get_likelihood_prob_x(l_train_x_dat, l_train_y_dat)

    def predict(self, l_test_x_dat):
        l_y = []
        l_y_prob = []

        for x_row in l_test_x_dat:
            y_prob = {}

            

            for o_y in self.lo_y:
                y_prob[o_y["y"]] = o_y["prior_prob"] 

            
            for i,x  in enumerate(x_row):
                if self.lo_x[i]["type"] == "discrete":
                    for j, o_y in enumerate(self.lo_y):
                        col_index = self.lo_x[i]["likely_info"]["col"].index(x)
                        y_prob[o_y["y"]] *= self.lo_x[i]["likely_info"]["likely_prob_mat"][j][col_index]
                else:
                    for j, o_y in enumerate(self.lo_y):
                   
                        mean = self.lo_x[i]["likely_info"]["norm_parm"][o_y["y"]]["mean"]
                        sigma2 = self.lo_x[i]["likely_info"]["norm_parm"][o_y["y"]]["sample_var"]
                        likely_prob = 1.0 /(2 * 3.14 * sigma2 ) * np.exp(-(x - mean)**2/(2*sigma2))

                        y_prob[o_y["y"]] *= likely_prob


            predict_y = self.lo_y[0]["y"]
            max_prob = y_prob[self.lo_y[0]["y"]]
            for i in range(1, len(self.lo_y)):
                if y_prob[self.lo_y[i]["y"]] > max_prob:
                    predict_y = self.lo_y[i]["y"]
                    max_prob = y_prob[self.lo_y[i]["y"]]
            
            l_y.append(predict_y)
            l_y_prob.append(y_prob)

        return l_y, l_y_prob
        

                    
                    

         

    def get_bayes_parm(self):
        return
