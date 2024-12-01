# CRTP: Compile-Time-Reflection-Programming

[wiki](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)

```cpp
// The Curiously Recurring Template Pattern (CRTP)
template <class T>
class Base {
  // methods within Base can use template to access members of Derived
};
class Derived : public Base<Derived> {
  // ...
};
```

Why?

Polymorphic chaining

```cpp
// Base class
template <typename ConcretePrinter>
class Printer {
public:
  Printer(ostream& pstream) : m_stream(pstream) {}

  template <typename T>
  ConcretePrinter& print(T&& t) {
    m_stream << t;
    return static_cast<ConcretePrinter&>(*this);
  }

  template <typename T>
  ConcretePrinter& println(T&& t) {
    m_stream << t << endl;
    return static_cast<ConcretePrinter&>(*this);
  }

private:
  ostream& m_stream;
};

// Derived class
class CoutPrinter : public Printer<CoutPrinter> {
public:
  CoutPrinter() : Printer(cout) {}

  CoutPrinter& SetConsoleColor(Color c) {
    // ...
    return *this;
  }
};

// usage
CoutPrinter().print("Hello ").SetConsoleColor(Color.red).println("Printer!");
```

Polymorphic copy construction
多态复制构造

When using polymorphism, one sometimes needs to create copies of objects by the base class pointer. A commonly used idiom for this is adding a virtual clone function that is defined in every derived class. The CRTP can be used to avoid having to duplicate that function or other similar functions in every derived class.
使用多态性时，有时需要通过基类指针创建对象的副本。为此，常用的习惯用法是添加在每个派生类中定义的虚拟克隆函数。 CRTP 可用于避免在每个派生类中重复该函数或其他类似函数。

```cpp
// Base class has a pure virtual function for cloning
class AbstractShape {
public:
  virtual ~AbstractShape() = default;
  virtual std::unique_ptr<AbstractShape> clone() const = 0;
};

// This CRTP class implements clone() for Derived
template <typename Derived>
class Shape : public AbstractShape {
public:
  std::unique_ptr<AbstractShape> clone() const override {
    return std::make_unique<Derived>(static_cast<Derived const&>(*this));
  }

protected:
  // We make clear Shape class needs to be inherited
  Shape() = default;
  Shape(const Shape&) = default;
  Shape(Shape&&) = default;
};

// Every derived class inherits from CRTP class instead of abstract class

class Square : public Shape<Square> {};

class Circle : public Shape<Circle> {};

```
This allows obtaining copies of squares, circles or any other shapes by shapePtr->clone().
这允许通过shapePtr->clone()获取正方形、圆形或任何其他形状的副本。