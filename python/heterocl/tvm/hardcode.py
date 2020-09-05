import os

def hardcode(filename):
    code = open(filename,"r").readlines()
    new_code, const = move_const(code)
    with open(filename,"w") as outfile:
        outfile.write(new_code)
    with open(os.path.join("/".join(filename.split("/")[:-1]),"const.h"),"w") as const_file:
        const_file.write(const)
    print("Done hardcode!")

def move_const(code):
    const = []
    res = []
    for line in code:
        if "const" in line:
            const.append(line.strip()+"\n")
        else:
            res.append(line)
    res.insert(11,'#include "const.h"\n\n')
    return "".join(res), "".join(const)