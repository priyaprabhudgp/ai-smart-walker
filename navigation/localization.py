class Localization:
    def __init__(self, start_location):
        self.current_location = start_location

    def update_location(self, new_location):
        self.current_location = new_location

    def get_location(self):
        return self.current_location