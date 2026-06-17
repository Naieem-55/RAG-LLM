class Student():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def display_info(self):
        print(f"Name: {self.name}, Age: {self.age}")

if __name__ == "__main__":
    student1 = Student("Alice", 20)
    student2 = Student("Bob", 22)

    student1.display_info()
    student2.display_info()