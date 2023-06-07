import random

class Thief:

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"Thief {self.id}"
    
class Cop:

    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"Cop {self.id}"
    
t1 = Thief(1)
t2 = Thief(2)
t3 = Thief(3)

c1 = Cop(1)
c2 = Cop(2)
c3 = Cop(3)

ts = [t1, t2, t3]
cs = [c1, c2, c3]

agents = random.sample(ts, len(ts)) + random.sample(cs, len(cs))
print(agents)
