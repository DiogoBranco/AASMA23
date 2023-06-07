class Entity:

    def __init__(self, id):
        self.id = id
        self.x = -1
        self.y = -1

    def set(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return type(self).__name__[0]
