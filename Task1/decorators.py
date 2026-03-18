def changecase(func):
    def myinner():
        return func().upper()
    return myinner


@changecase
def myfunction():
    return "hello"


@changecase
def myfunction2():
    return "world"


print(myfunction())
print(myfunction2())
