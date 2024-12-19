# ArgParse in Python

## unexpected `parser.add_argument("--train", type=bool)`

在使用 `argparse` 时，`parser.add_argument("--train", type=bool)` 的用法可能会导致一些意外行为，因为 `type=bool` 并不会将输入的字符串自动转换为布尔值。相反，它会尝试将输入的字符串作为 Python 的 `bool()` 函数的参数，这通常会导致输入的值被解释为 `True` 或 `False`，但结果可能不符合预期。

### 问题分析

- `bool("True")` 返回 `True`，但 `bool("False")` 也返回 `True`，因为非空字符串在 Python 中被视为 `True`。
- 因此，直接使用 `type=bool` 无法正确解析布尔值参数。

### 正确的用法

为了正确解析布尔值参数，可以使用 `action='store_true'` 或 `action='store_false'`，或者自定义类型转换函数。

#### 方法 1：使用 `action='store_true'` 或 `action='store_false'`

这是最常用的方法，适用于布尔值参数的默认行为。

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="是否启用训练模式")

args = parser.parse_args()

print(f"训练模式: {args.train}")
```

##### 使用示例

```bash
python script.py --train
```

输出：

```
训练模式: True
```

如果不传递 `--train` 参数：

```bash
python script.py
```

输出：

```
训练模式: False
```

#### 方法 2：自定义类型转换函数

如果你希望用户明确传递 `True` 或 `False`，可以使用自定义类型转换函数。

```python
import argparse

def str_to_bool(value):
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise argparse.ArgumentTypeError("布尔值必须是 'True' 或 'False'")

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str_to_bool, help="是否启用训练模式")

args = parser.parse_args()

print(f"训练模式: {args.train}")
```

##### 使用示例

```bash
python script.py --train True
```

输出：

```
训练模式: True
```

```bash
python script.py --train False
```

输出：

```
训练模式: False
```

如果传递了无效的值：

```bash
python script.py --train maybe
```

输出：

```
usage: script.py [-h] [--train TRAIN]
script.py: error: argument --train: 布尔值必须是 'True' 或 'False'
```

### 总结

- 如果希望参数默认是布尔值，推荐使用 `action='store_true'` 或 `action='store_false'`。
- 如果需要用户明确传递 `True` 或 `False`，可以使用自定义类型转换函数。
- 直接使用 `type=bool` 可能会导致意外行为，不推荐使用。
