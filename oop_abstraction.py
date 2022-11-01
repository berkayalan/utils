from abc import ABC, abstractmethod

class Employee(ABC):

    def __init__(self, name, surname, salary,department) -> None:
        super().__init__()
        self.name = name
        self.surname = surname
        self.salary = salary
        self.department = department
    
    def employee_information(self):
        return "Employee name is" +str(self.name)+ "and this employee is in" +self.department+" department."

    @abstractmethod
    def change_department(self, new_department):
        self.department = new_department
        print("Employee's new department is ",new_department)
    
    @abstractmethod
    def salary_increase(self, ratio):
        self.salary += ((self.salary*ratio)/100)

class Marketing(Employee):
    def __init__(self, name, surname, salary,department="Marketing") -> None:
        super().__init__(name, surname, salary,department)

    def change_department(self, new_department):
        return super().change_department(new_department)
    
    def salary_increase(self, ratio):
        return super().salary_increase(ratio)

class Data(Employee):
    def __init__(self, name, surname, salary,story_point,department="Data") -> None:
        super().__init__(name, surname, salary,department)
        self.story_point = story_point
    
    def create_new_task(self,new_task,story_point):
        print("New task is", new_task)
        self.story_point+=story_point
        print("Story points increased to ",self.story_point)

    def change_department(self, new_department):
        return super().change_department(new_department)
    
    def salary_increase(self, ratio):
        return super().salary_increase(ratio)

m1 = Marketing("Berkay" , "Alan", 5000)
print(m1.department)
print(m1.name)
print(m1.salary)

try:
    m2 = Employee("Berkay" , "Alan", 5000,"Data")
except TypeError as e:
    print("This is a abstract class and can't instantiate!")

d1 = Data("Berkay" , "Alan",story_point = 40, salary = 15000)
print("Department: ",d1.department)
print("Name: ",d1.name)
print("Salary: ",d1.salary)
print("Story Point: ", d1.story_point)


