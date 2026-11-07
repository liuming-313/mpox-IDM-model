import random

class Person:
    def __init__(self, uid, age, gender, sexual_orientation):
        self.uid = uid
        self.age = age
        self.gender = gender
        self.sexual_orientation = sexual_orientation
        self.partners = []  # list of uid of partners
        self.infected = False
        self.infectious = False
        self.recovered = False

    def __repr__(self):
        status = 'I' if self.infected else 'S'
        return f"Person({self.uid}, {self.gender}, {self.sexual_orientation}, age={self.age}, status={status})"
