class BankAccount:
    def __init__(self, account_number, balance=0):
        self._account_number = account_number
        self._balance = balance

    def deposit(self, amount):
        if amount >= 100:
            self._balance += amount
            print(f"Deposited LKR {amount}. New Balance: LKR {self._balance}")
        else:
            print("Deposit is not successful")

    def withdraw(self, amount):
        if amount <= self._balance:
            self._balance -= amount
            print(f"withdarw LKR {amount}. New Balance: LKR {self._balance}")
        else:
            print("Withdrawal is not successful")


class SavingsAccount(BankAccount):
    def __init__(self, account_number, balance=0, interest_rate=0.05):
        super().__init__(account_number, balance)
        self.interest_rate = interest_rate

    def calculate_interest(self):
        interest = self._balance * self.interest_rate
        total_amount = self._balance + interest
        print(f"Interest: LKR {interest}")
        print(f"Total Amount: LKR {total_amount}")


def total_balance(amount, total_amount):
    total_balance = amount + total_amount
    print(f"Total Balance: LKR {total_balance}")


account = BankAccount("123456789")
account.deposit(1000)
account.deposit(800)
account.deposit(300)
account.withdraw(500)
account_savings = SavingsAccount("987654321", balance=1000)
account_savings.calculate_interest()
total_balance(account._balance, account_savings._balance)
