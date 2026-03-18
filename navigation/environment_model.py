class EnvironmentModel:
    def __init__(self):
        # Graph: node -> connected nodes
        self.map = {}

    def add_location(self, location):
        if location not in self.map:
            self.map[location] = []

    def connect_locations(self, loc1, loc2):
        self.map.setdefault(loc1, []).append(loc2)
        self.map.setdefault(loc2, []).append(loc1)

    def get_neighbors(self, location):
        return self.map.get(location, [])

    def __str__(self):
        return str(self.map)