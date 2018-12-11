"""
Machine Learning Project 4
COSC-247, Taught by Professor Alfeld

By James Corbett and Juhwan Jeong
December 2018
"""

import numpy as np
import sklearn
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import ensemble
from sklearn import metrics
from sklearn import multioutput

# Comment/uncomment the following 3 lines if you're getting annoyed with warnings from sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)




class LatLonPair:
    """
    Represents latitude and longitude as a pair.
    """
    def __init__(self, latitude=0.00, longitude=0.00):
        self.latitude = latitude
        self.longitude = longitude


class DataBuilder:
    """
    Takes in three text files containing training data, test data, and a file representing connections between users. From these, this class builds
    data matrices which can be passed to sklearn models.
    """

    def __init__(self, train_path=None, test_path=None, graph_path=None, out_path=None, lift_degree=1, use_mode=True, \
        mode_param=0.088, replace_missing_values=True, exclude_null_island=True, include_posts=False, avg_post_hr=-1):
        """
        Builds the data sets.

        :param train_path:                  Path to a text file containing the data for the training set, containing entries 
                                            on each line like "Id,Hour1,Hour2,Hour3,Lat,Lon,Posts\n".
                                            The first line should not contain data.
        :param test_path:                   Path to a text file containing the data for the test set, containing entries 
                                            on each line like "Id,Hour1,Hour2,Hour3,Posts\n".
                                            The first line should not contain data.   
        :param graph_path:                  Path to a text file containing the connections between users, formatted like "user_ID_1\tuser_ID_2". 
                                            The data should begin on the first line.                                           
        :param out_path:                    Name of the text file you want to create that contains the results
        :param lift_degree:                 Degree that you want to lift your data
        :param use_mode:                    Include the mode in the training and test sets
        :param mode_param:                  The "size" of units in the mode
        :param replace_missing_values:      Replace 0's with the average of that feature space
        :param exclude_null_island:         Exclude points with (lat,lon) == (0,0) 
        :param include_posts:               Include number of posts as a feature for each data point
        :param avg_post_hr:                 What you think the average hour that people post on social media. Used for translating post hour to longitude.
                                            
        :Returns:            An object with fields containing the training set and test set
        """

        self.mode = use_mode
        self.mode_param = mode_param
        self.lift_degree = lift_degree
        self.out_path = out_path
        self.exclude_null_island = exclude_null_island
        self.avg_post_hr = avg_post_hr
        self.include_posts = include_posts
        self.replace_missing_values = replace_missing_values
        self.graph = self.read_graph(graph_path)
        self.raw_train_data = self.read_in_data(train_path)
        self.user_locations = self.set_locations()
        self.X_tr, self.y_tr_lat, self.y_tr_lon, self.y_both = self.build_training_set()
        self.test_set, self.user_ids = self.build_test_set(test_path)
        
        #self.print_data(self.X_tr.tolist(), 10) #65
        #self.print_data(self.test_set.tolist(), 20)
        self.lat_preds = None
        self.lon_preds = None
        self.both_preds = None


        self.fit_and_predict()



    def read_graph(self, filename):
        """
        Reads connections from the given file and creates a mapping of user_id to that user's connections.
        """
        friend_graph = {}
        with open(filename, "r") as file:
            for line in file:
                data = line.split("\t", 2)
                user_id = int(data[0])
                friend_id = int(data[1])
                if user_id not in friend_graph:
                    friend_graph[user_id] = [friend_id]
                else:
                    friend_graph[user_id].append(friend_id) 
                if friend_id not in friend_graph:
                    friend_graph[friend_id] = [user_id]
                else:
                    friend_graph[friend_id].append(user_id)                          
        return friend_graph

    
   
    def read_in_data(self, filename):
        """
        Reads in the training set and represents as a 2D list. The methods set_locations() and build_training_set() then extract data from the list.
    
        Takes in the path of the training set text file as a parameter.
        """
        data = []
        with open(filename, "r") as file:
            file.readline()           
            for line in file:
                data_string = line.split(",", 7)
                data_point = [int(data_string[i]) for i in range(0,4)] + [float(data_string[i]) for i in range(4,6)] + [int(data_string[6])]
                data.append(data_point)
        return data



    def set_locations(self):
        """
        Creates a dictionary which maps user IDs to user locations.
        """
        map = {}
        data = self.raw_train_data
        for data_point in data:
            user_id = data_point[0]
            user_location = LatLonPair(data_point[4], data_point[5])
            map[user_id] = user_location
        return map        




    def build_training_set(self):
        """
        Builds the training set, which is the pair (X_tr, y_tr).
        Each point in the training set X_tr has features (hour1, hour2, hour3, posts, average latitude of friends, average longitude of friends, 
        (and maybe also mode_lat, mode_lon)
        Swaps 25 with 0.
        """
        X_tr = []
        y_tr_lat = []
        y_tr_lon = []
        y_both = []
        data = self.raw_train_data
        for data_point in data:
            user_id = data_point[0]
            if data_point[6] < 0:
                print("alert!")
            if data_point[4] is not 0.00 or data_point[5] is not 0.00 or self.exclude_null_island is False:
                for i in range(1, 4):
                    if data_point[i] is 25:
                        data_point[i] = 0
                    else:
                        data_point[i] += 1                    
                X_point = [int(data_point[i]) for i in range(1,4)]
                if self.include_posts is True:
                    X_point = X_point + [int(data_point[6])]
                X_point = X_point + self.get_avg_lat_lon(user_id) 
                if self.mode is True:
                    X_point = X_point + self.get_mode_lat_lon(user_id, self.mode_param)
                X_tr.append(X_point)
                y_tr_lat.append(data_point[4])
                y_tr_lon.append(data_point[5])
                y_both.append([data_point[4], data_point[5]]) 
            else: 
                continue    
        
        if self.replace_missing_values is True:
            self.replace_missing_values_data(X_tr)
        if self.avg_post_hr > -1:
            for k in range(0, len(X_tr)):
                hours_to_longitude = [(-15)*(X_tr[k][i]-self.avg_post_hr) for i in range(0,3)]
                j = 0
                for x in hours_to_longitude:
                    if x < -180:
                        x += 180
                        hours_to_longitude[j] = -x
                    elif x >= 180:
                        x -= 180
                        hours_to_longitude[j] = -x    
                    j+=1
                
                avg_hour = sum(hours_to_longitude) / len(hours_to_longitude)
                #X_tr[k] = X_tr[k] + hours_to_longitude
                X_tr[k].append(avg_hour)
        return (np.array(X_tr), np.array(y_tr_lat), np.array(y_tr_lon), np.array(y_both))

    def build_test_set(self, filename):
        """
        Builds the test set. Swaps 25 with 0.
        """
        to_return = []
        users = []
        with open(filename, "r") as file:
            file.readline()         
            for line in file:
                data = line.split(",", 5)
                user_id = int(data[0])
                for i in range(1, 4):
                    data[i] = int(data[i])
                    if data[i] is 25:
                        data[i] = 0
                    else:
                        data[i] += 1 
                users.append(user_id)
                data_point = [int(data[i]) for i in range(1,4)] 
                if self.include_posts is True:
                    data_point = data_point + [int(data[4])]
                data_point = data_point + self.get_avg_lat_lon(user_id)
                if self.mode is True:
                    data_point = data_point + self.get_mode_lat_lon(user_id, self.mode_param)
                to_return.append(data_point)
        
        if self.replace_missing_values is True:
            self.replace_missing_values_data(to_return)
        if self.avg_post_hr > -1:
            for k in range(0, len(to_return)):
                hours_to_longitude = [(-15)*(to_return[k][i]-self.avg_post_hr) for i in range(0,3)]
                j = 0
                for x in hours_to_longitude:
                    if x < -180:
                        x += 180
                        hours_to_longitude[j] = -x
                    elif x >= 180:
                        x -= 180
                        hours_to_longitude[j] = -x    
                    j+=1
                
                avg_hour = sum(hours_to_longitude) / len(hours_to_longitude)
                to_return[k].append(avg_hour)
        return (np.array(to_return), users) 

   

    def get_avg_lat_lon(self, user_id):
        """
        Takes a user ID number and returns the average latitude and longitude of their connections/friends as a list [avg_lat, avg_lon].
        Returns [0.00, 0.00] if user has no connections.
        """
        if user_id not in self.graph:
            return [0.00, 0.00]
        else:    
            list_of_connections = self.graph[user_id]
            locations = self.user_locations
            total_lat = 0.00
            total_lon = 0.00
            count = 0
            for friend_id in list_of_connections:
                if friend_id in locations:
                    count += 1
                    friend_location = locations[friend_id]
                    total_lat += friend_location.latitude
                    total_lon += friend_location.longitude
                else:
                    continue
            if count > 0:
                return [total_lat/count, total_lon/count]
            else:
                return [0.00, 0.00]


    def get_mode_lat_lon(self, user_id, m):
        """
        Takes a user ID and returns the mode latitude and longituide of all the user's friends
        Returns [0.00, 0.00] if user has no connections.
        Classifies location by rounding long/lat to nearest multiple of m
        :param self:
        :param user_id:
        :return:
        """
        if user_id not in self.graph:
            return [0.00, 0.00]
        list_of_connections = self.graph[user_id]
        locations = self.user_locations
        freq_of_locations = {}
        max_freq = 0
        for friend_id in list_of_connections:
            if friend_id in locations:
                friend_location = locations[friend_id]
                lat = int(friend_location.latitude / m) * m
                long = int(friend_location.longitude / m) * m
                if friend_location.latitude - lat >= m / 2:
                    lat += m
                if friend_location.longitude - long >= m / 2:
                    long += m
                if (lat, long) not in freq_of_locations:
                    freq_of_locations[(lat, long)] = 1
                else:
                    freq_of_locations[(lat, long)] += 1
                if freq_of_locations[(lat, long)] > max_freq:
                    max_freq = freq_of_locations[(lat, long)]

        # if multiple nodes are found, take the average
        avg_lat = 0
        avg_long = 0
        count = 0
        for location_pair in freq_of_locations:
            if freq_of_locations[location_pair] == max_freq:
                avg_lat += location_pair[0]
                avg_long += location_pair[1]
                count += 1
        if count == 0:
            return [0,0]
        return [avg_lat / count, avg_long / count]

    
    def print_txt(self):
        """
        Prints the predictions to a text file in the format (user_id,lat_pred,lon_pred)
        """
        f = open(self.out_path, "x")
        f.write("Id,Lat,Lon\n")
        if self.both_preds is not None:
            for user_id, pair in zip(self.user_ids, self.both_preds):
                f.write("{0},{1},{2}\n".format(user_id, pair[0], pair[1]))
        else:
            for user_id, lat_pred, lon_pred in zip(self.user_ids, self.lat_preds, self.lon_preds):
                f.write("{0},{1},{2}\n".format(user_id, lat_pred, lon_pred))
        f.close()    


    def lift_data(self, train_data, test_data):
        """
        Lifts data to the degree specified in the __init__() function
        """
        lifter = sklearn.preprocessing.PolynomialFeatures(degree=self.lift_degree)

        lifted_train_data = lifter.fit_transform(train_data)
        lifted_test_data = lifter.transform(test_data)
        return (lifted_train_data, lifted_test_data)

    def standardize_data(self, train_data, test_data):
        """
        Uses a StandardScaler to preprocess training and testing data
        """
        stdizer = sklearn.preprocessing.StandardScaler()
        train_data = stdizer.fit_transform(train_data)
        test_data = stdizer.transform(test_data)
        return (train_data, test_data)

    def MinMax_scale_data(self, train_data, test_data):
        """
        Uses a MinMaxScaler to preprocess training and testing data
        """
        scaler = sklearn.preprocessing.MinMaxScaler()
        train_data = stdizer.fit_transform(train_data)
        test_data = stdizer.transform(test_data)
        return (train_data, test_data)

    def cross_val(self, models, X_tr, y_tr, folds=10, scorer="neg_mean_squared_error"):
        """
        Takes a bunch of models, performs cross validation on them and takes the best model. Prints results. 

        :param models:  A list of models to try
        :param X_tr:    The set to train the models on (might be lifted data, for instance)
        :param y_tr:    The target values accompanying y_tr
        :param folds:   Number of folds for cross-validation
        :scorer:        The scoring function to use
        """
        max = -np.inf
        best_model = None
        for model in models:
            score = sklearn.model_selection.cross_val_score(model, X_tr, y_tr, scoring=scorer, cv=folds)
            avg_score = sum(score) / float(len(score))
            if avg_score>max:
                max = avg_score
                best_model = model
                print("best model updating to:")
                print(str(best_model))
                print(str(score))
                print(str(avg_score))
            else:
                print("This model is not the best")
                print(str(model))
                print(str(score))
                print(str(avg_score))
        print("best model is" + str(best_model))
        return best_model        


    def print_data(self, matrix, count):
        """
        Prints lines of data matrix. Useful for checking on how yoour features compare to the actual values

        :param matrix:  The matrix to print from
        :param count:   The number of lines to print
        """
        print("hour1,hour2,hour3,posts,avg_lat,avg_lon,\tactual_lat,actual_lon")
        n = 0
        print(len(matrix[0]))
        for line, lat, lon in zip(matrix, self.y_tr_lat, self.y_tr_lon):
            n+=1
            if n is count:
                return
            else:    
                print(str(line)+",\t"+str(lat)+","+str(lon))        


    def replace_missing_values_data(self, data_matrix):
        """
        Replaces points in the data set with value 0 (here representing insufficient information) with the average of that feature space.

        :param data_matrix:     The matrix to perform replacement on
        """
        n_features = 5
        if self.mode is True:
            n_features += 2
        if self.exclude_null_island is False:
            n_features += 1   

        n_features = len(data_matrix[0])     

        tot = [0 for i in range(n_features)]
        count = [0 for i in range(n_features)]

        for row in data_matrix:
            for i in range(len(row)):
                if row[i] != 0:
                    tot[i] += row[i]
                    count[i] += 1

        for row in data_matrix:
            for i in range(len(row)):
                if row[i] == 0:
                    row[i] = tot[i]/count[i]



    def fit_and_predict(self):
        """
        Insert your models here to do the work of predicting on the training set.
        """
        

        lifted_train_data, lifted_test_data = self.lift_data(self.X_tr, self.test_set)
        lifted_train_data, lifted_test_data = self.standardize_data(lifted_train_data, lifted_test_data)
        self.X_tr, self.test_set = self.standardize_data(self.X_tr, self.test_set)


        #Here are some good models, which you might want to use (they're not quite as good as GradientBoostingRegressors)
        #the models are multi_Ada, multi_ridge_chain, and tree.
        #All of them predict both latitude and longitude at the same time, rather than one and then the other
        #Use multioutput.MultiOutputRegressor() or multioutput.RegressorChain() to get multiple predictions at once
        #Tree models don't need to be fit on lifted data. Only ridge needs to be fit on lifted data.
        #multi_grad is a GradientBoostingRegressor that does pretty well. Yours might be better, though.
        ridge = linear_model.Ridge(alpha=2.0, tol=0.0001)
        multi_ridge_chain = multioutput.RegressorChain(base_estimator=ridge, order=[0,1])
        tree = sklearn.tree.DecisionTreeRegressor(max_depth=9, min_samples_split=200, min_samples_leaf=100)

        Ada_tree = sklearn.tree.DecisionTreeRegressor(max_depth=2, min_samples_split=100, min_samples_leaf=50)
        AdaBoost = ensemble.AdaBoostRegressor(base_estimator=Ada_tree, learning_rate=0.02, n_estimators=60)
        multi_Ada = multioutput.MultiOutputRegressor(estimator=AdaBoost, n_jobs=3)

        gradient_boost = ensemble.GradientBoostingRegressor(learning_rate = 0.065, n_estimators=500)
        multi_grad = multioutput.MultiOutputRegressor(estimator=gradient_boost, n_jobs=3)

        self.print_txt()
        



dataBuilder = DataBuilder(train_path="posts_train.txt", test_path="posts_test.txt", \
                          graph_path="graph.txt", out_path="final2.txt", \
                          lift_degree=4, use_mode=True, mode_param=0.088, replace_missing_values=True, \
                          exclude_null_island=False, include_posts=True, avg_post_hr=20)
