# load_code.py
def load_code(file_path):
    with open(file_path, 'r') as file:
        return f"```python\n{file.read()}\n```"

if __name__ == "__main__":
    code_block = load_code("xx.py")
    with open("README.md", 'a') as md_file:
        md_file.write("\n" + code_block)