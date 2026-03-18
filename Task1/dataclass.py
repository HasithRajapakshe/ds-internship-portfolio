from dataclasses import dataclass


@dataclass
class Person:
    name: str
    age: int


person1 = Person(name="Alice", age=30)
print(person1)
