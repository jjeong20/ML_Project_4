"""
Machine Learning Project 4
COSC-247, Taught by Professor Alfeld

By James Corbett and Juhwan Jeong
December 2018
"""

import numpy as np
from sklearn import kernel_ridge
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor


# import sys


class DataBuilder:
    """
    Takes in three text files containing training data, test data, and a file representing connections between users. From these, this class builds
    data matrices which can be passed to sklearn models.
    """

    def __init__(self, train_path=None, test_path=None, graph_path=None, margin=0.01):
        """
        Builds the data sets.

        @param train_path:  Path to a text file containing the data for the training set, containing entries
                            on each line like "Id,Hour1,Hour2,Hour3,Lat,Lon,Posts\n".
                            The first line should not contain data.
        @param test_path:   Path to a text file containing the data for the test set, containing entries
                            on each line like "Id,Hour1,Hour2,Hour3,Posts\n".
                            The first line should not contain data.
        @param graph_path:  Path to a text file containing the connections between users, formatted like "user_ID_1\tuser_ID_2".
                            The data should begin on the first line.

        Returns:            An object with fields containing the training set and test set
        """
        self.graph = self.read_graph(graph_path)
        self.raw_train_data = self.read_in_data(train_path)
        self.user_locations = self.set_locations()
        self.X_tr, self.y_tr_lat, self.y_tr_lon = self.build_training_set(margin)
        self.test_set, self.user_ids = self.build_test_set(test_path)
        self.lat_preds = None
        self.lon_preds = None

        scaler = MinMaxScaler()

        scaler.fit(self.X_tr)
        MinMaxScaler(copy=True, feature_range=(0, 1))
        self.X_tr_scaled = scaler.transform(self.X_tr)
        # self.test_set_scaled = scaler.transform(self.test_set)

    def read_graph(self, filename):
        friend_graph = {}
        count = 0
        with open(filename, "r") as file:
            for line in file:
                count += 1
                data = line.split("\t", 2)
                user_id = int(data[0])
                friend_id = int(data[1])
                if user_id not in friend_graph:
                    friend_graph[user_id] = [friend_id]
                elif friend_id in friend_graph[user_id]:
                    print("Error in line {0}! This edge between {1} and {2} has already been found.".format(count,
                                                                                                            user_id,
                                                                                                            friend_id))
                    # sys.exit()
                else:
                    friend_graph[user_id].append(friend_id)
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
                data_point = [int(data_string[i]) for i in range(0, 4)] + [float(data_string[i]) for i in
                                                                           range(4, 6)] + [int(data_string[6])]
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

    def build_training_set(self, margin):
        """
        Builds the training set, which is the pair (X_tr, y_tr) Also returns user list.
        Each point in the training set X_tr has features (hour1, hour2, hour3, posts, average latitude of friends, average longitude of friends)
        """
        X_tr = []
        y_tr_lat = []
        y_tr_lon = []
        # users = []
        data = self.raw_train_data
        for data_point in data:
            user_id = data_point[0]
            if data_point[4] is not 0.00 or data_point[5] is not 0.00:
                for i in range(1, 4):
                    if data_point[i] is 25:
                        data_point[i] = 0
                    else:
                        data_point[i] += 1

                    X_point = [int(data_point[i]) for i in range(1, 4)] + [
                        int(data_point[6])] + self.get_avg_lat_lon(
                        user_id) + self.get_mode_lat_long(user_id,
                                                          margin)  # hour1,hour2,hour3,posts,avg_lat,avg_lon, mode lat, mode long in that order

                X_tr.append(X_point)
                y_tr_lat.append(data_point[4])
                y_tr_lon.append(data_point[5])
                # y_point = [data_point[4], data_point[5]]    #latitude and longitude
                # y_tr.append(y_point)
                # users.append(int(data_point[0]))
            else:
                continue
        return (np.array(X_tr), np.array(y_tr_lat), np.array(y_tr_lon))

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
                return [total_lat / count, total_lon / count]
            else:
                return [0.00, 0.00]

    def build_test_set(self, filename):
        """
        Builds the test set.
        """
        to_return = []
        users = []
        with open(filename, "r") as file:
            file.readline()
            for line in file:
                data = line.split(",", 5)
                user_id = int(data[0])
                users.append(user_id)
                data_point = [int(data[i]) for i in range(1, 5)] + self.get_avg_lat_lon(user_id)
                to_return.append(data_point)
        return (np.array(to_return), users)

    def print_txt(self):
        """
        Prints the predictions to a text file in the
        """
        f = open("results.txt", "x")
        f.write("Id,Lat,Lon\n")
        for user_id, lat_pred, lon_pred in zip(self.user_ids, self.lat_preds, self.lon_preds):
            f.write("{0},{1},{2}\n".format(user_id, lat_pred, lon_pred))
        f.close()

    def get_mode_lat_long(self, user_id, m):
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
            return [0, 0]
        return [avg_lat / count, avg_long / count]


class LatLonPair:
    """
    Represents latitude and longitude as a pair.
    """

    def __init__(self, latitude=0.00, longitude=0.00):
        self.latitude = latitude
        self.longitude = longitude


data = {}
for i in range(1, 101):
    data[i / 1000] = DataBuilder("posts_train.txt", "posts_test.txt", "graph.txt", i / 1000)
    print(i / 1000)

score_to_data = {}

for i in data:
    d = data[i]
    clf1 = GridSearchCV(linear_model.LinearRegression(), {'normalize': [False]}, cv=10,
                        scoring='neg_mean_squared_error')
    clf1.fit(d.X_tr_scaled, d.y_tr_lat)
    means = clf1.cv_results_['mean_test_score']
    score_to_data[means[0]] = "linear " + str(i)
    print('linear done')

    clf2 = GridSearchCV(linear_model.Ridge(), {}, cv=10, scoring='neg_mean_squared_error')
    clf2.fit(d.X_tr_scaled, d.y_tr_lat)
    means = clf2.cv_results_['mean_test_score']
    score_to_data[means[0]] = "ridge " + str(i)
    print('ridge done')
    """
    clf3 = GridSearchCV(kernel_ridge.KernelRidge(), {}, cv=10, scoring='neg_mean_squared_error')
    clf3.fit(d.X_tr_scaled, d.y_tr_lat)
    means = clf3.cv_results_['mean_test_score']
    score_to_data[means[0]] = "kernel ridge " + str(i)
    print('kernel ridge done')

    clf4 = GridSearchCV(SVR(),
                        {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}, cv=10, scoring='neg_mean_squared_error',
                        n_jobs=-1)
    clf4.fit(d.X_tr_scaled, d.y_tr_lat)
    means = clf4.cv_results_['mean_test_score']
    score_to_data[means[0]] = "svr " + clf4.best_params + str(i)
    print('svr done')
    """
    clf5 = GridSearchCV(linear_model.SGDRegressor(), {}, cv=10, scoring='neg_mean_squared_error')
    clf5.fit(d.X_tr_scaled, d.y_tr_lat)
    means = clf5.cv_results_['mean_test_score']
    score_to_data[means[0]] = "gradient descent " + str(i)
    print('gradient descent done')
    """
    clf6 = GridSearchCV(GaussianProcessRegressor(), {}, cv=10, scoring='neg_mean_squared_error')
    clf6.fit(d.X_tr_scaled, d.y_tr_lat)
    means = clf6.cv_results_['mean_test_score']
    score_to_data[means[0]] = "gaussian " + str(i)
    print('gaussian done')
    """

print(score_to_data)
min_score = min(score_to_data)
print(min_score)
print(score_to_data[min_score])


"""
model_lat = linear_model.LinearRegression()
model_lat.fit(dataBuilder.X_tr, dataBuilder.y_tr_lat)
dataBuilder.lat_preds = model_lat.predict(dataBuilder.test_set)

model_lon = linear_model.LinearRegression()
model_lon.fit(dataBuilder.X_tr, dataBuilder.y_tr_lon)
dataBuilder.lon_preds = model_lon.predict(dataBuilder.test_set)
"""
"""
model_lat = kernel_ridge.KernelRidge()
model_lat.fit(dataBuilder.X_tr, dataBuilder.y_tr_lat)
dataBuilder.lat_preds = model_lat.predict(dataBuilder.test_set)

model_lon = kernel_ridge.KernelRidge()
model_lon.fit(dataBuilder.X_tr, dataBuilder.y_tr_lon)
dataBuilder.lon_preds = model_lon.predict(dataBuilder.test_set)
"""
