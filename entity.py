class Entity:

    def __init__(self, id : int, x : int, y : int):
        self.id = id
        self.x = x
        self.y = y

    def __str__(self):
        return type(self).__name__[0]
