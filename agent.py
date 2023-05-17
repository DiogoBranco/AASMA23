class Agent:
    def __init__(self, x, y, field_of_view):
        self.x = x
        self.y = y
        self.field_of_view = field_of_view

class Cop(Agent):
    def __init__(self, x, y, field_of_view=3):
        super().__init__(x, y, field_of_view)

class Thief(Agent):
    def __init__(self, x, y, field_of_view=3):
        super().__init__(x, y, field_of_view)