class person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age


p1 = person("John", 30)
print(p1._person__name)
print(p1._person__age)
