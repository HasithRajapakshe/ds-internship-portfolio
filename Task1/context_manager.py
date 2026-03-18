class File:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        print(f'{self.name} opened')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f'{self.name} closed')


if __name__ == '__main__':
    with File('file.txt') as f:
        print(f'{f.name} is being processed')
