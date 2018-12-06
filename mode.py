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
            lat = int(friend_location.latitude/m)*m
            long = int(friend_location.longitude/m)*m
            if friend_location.latitude - lat >= m/2:
                lat += m
            if friend_location.longitude - long >= m/2:
                long += m
            if [lat, long] not in freq_of_locations:
                freq_of_locations[[lat, long]] = 1
            else:
                freq_of_locations[[lat, long]] += 1
            if freq_of_locations[[lat, long]] > max_freq:
                max_freq = freq_of_locations

    #if multiple nodes are found, take the average
    avg_lat = 0
    avg_long = 0
    count = 0
    for location_pair in freq_of_locations:
        if freq_of_locations[location_pair] == max_freq:
            avg_lat += location_pair[0]
            avg_long += location_pair[1]
    return [avg_lat/count, avg_long/count]
