"""
Machine Learning Project 4
COSC-247, Taught by Professor Alfeld

By James Corbett and Juhwan Jeong
December 2018
"""

import numpy as np
#import sys





class DataBuilder:
    """
    Takes in three text files containing training data, test data, and a file representing connections between users. From these, this class builds
    data matrices which can be passed to sklearn models.
    """

    def __init__(self, train_path=None, test_path=None, graph_path=None):
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
        self.X_tr, self.y_tr = self.build_training_set()
        self.test_set = self.build_test_set(test_path)



    def read_graph(self, filename):
        friend_graph = {}
        count = 0
        with open(filename, "r") as file:
            for line in file:
                count+=1
                data = line.split("\t", 2)
                user_id = int(data[0])
                friend_id = int(data[1])
                if user_id not in friend_graph:
                    friend_graph[user_id] = [friend_id]
                elif friend_id in friend_graph[user_id]:
                    print("Error in line {0}! This edge between {1} and {2} has already been found.".format(count, user_id, friend_id)) 
                    #sys.exit()   
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
        """
        X_tr = []
        y_tr = []
        data = self.raw_train_data
        for data_point in data:
            user_id = data_point[0]
            X_point = [int(data_point[i]) for i in range(1,4)] + [int(data_point[6])] + self.get_avg_lat_lon(user_id)   #hour1,hour2,hour3,posts,avg_lat,avg_lon in that order
            X_tr.append(X_point)
            y_point = [data_point[4], data_point[5]]    #latitude and longitude
            y_tr.append(y_point)
        return (np.array(X_tr), np.array(y_tr))

   

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

    

    def build_test_set(self, filename):
        """
        Builds the test set. 
        """
        to_return = []
        with open(filename, "r") as file:
            file.readline()			
            for line in file:
                data = line.split(",", 5)
                user_id = int(data[0])
                data_point = [int(data[i]) for i in range(1,5)] + self.get_avg_lat_lon(user_id)
                to_return.append(data_point)
        return np.array(to_return)	

    
    


class LatLonPair:
    """
    Represents latitude and longitude as a pair.
    """
    def __init__(self, latitude=0.00, longitude=0.00):
        self.latitude = latitude
        self.longitude = longitude




dataBuilder = DataBuilder("posts_train.txt", "posts_test.txt", "graph.txt")

