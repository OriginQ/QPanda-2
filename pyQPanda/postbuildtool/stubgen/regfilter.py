import re

def filterFunc(func_str: str):
    s = func_str.replace("pyQPanda.", "")
    s = s.replace("QPanda::", "")
    r = re.findall(r"<[\w\s\.]{0,}>", s)
    for i in r:
        s = s.replace(i, "...")
    return s
