class Vehicle:
    def __init__(self, brand, model, year):
        self.brand = brand
        self.model = model
        self.year = year

    def start_engine(self):
        print(f"{self.brand} {self.model} engine started.")


class Car(Vehicle):
    pass


class Motorcycle(Vehicle):
    def start_engine(self):
        print(f"{self.brand} {self.model} motorcycle engine started.")


class Truck(Vehicle):
    def start_engine(self):
        print(f"{self.brand} {self.model} truck engine started.")


car1 = Car("Toyota", "Camry", 2022)
car1.start_engine()
