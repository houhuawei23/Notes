from jinja2 import Template

# 定义模板
template = Template(open("test.jinja2").read())

# 渲染模板
output = template.render(target="RISCV")

# 打印生成的 HTML
print(output)
