import random

class SimpleSim:
    def __init__(self, pop, infection_prob=0.05):
        self.pop = pop
        self.day = 0
        self.infection_prob = infection_prob

    def seed_infection(self, n=1):
        initial = random.sample(self.pop.people, n)
        for p in initial:
            p.infected = True
            p.infectious = True
        print(f"Seeded {n} infections.")

    def step(self):
        new_infections = []

        for person in self.pop.people:
            if person.infectious:
                for pid in person.partners:
                    partner = self.pop.people[pid]
                    if not partner.infected and random.random() < self.infection_prob:
                        new_infections.append(partner)

        for p in new_infections:
            p.infected = True
            p.infectious = True

        self.day += 1
        print(f"Day {self.day}: {len(new_infections)} new infections.")

    def summary(self):
        total = len(self.pop)
        infected = len(self.pop.get_infected())
        print(f"Day {self.day}: {infected}/{total} people infected.")
