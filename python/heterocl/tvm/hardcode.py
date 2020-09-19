import os

def hardcode(filename):
    code = open(filename,"r").readlines()
    new_code, const = move_const(code)
    new_code = move_pipeline_inward(new_code)
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

def remove_interface(code):
    res = ""
    for i,line in enumerate(code.split("\n")):
        if not "#pragma HLS INTERFACE" in line:
            res += line +"\n"
    return res

def move_pipeline_inward(code):
    res = []
    idx = []
    for i,line in enumerate(code.split("\n")):
        if "conv1_ff" in line or "conv2_ff" in line:
            idx.append(i)
        res.append(line)
    for i in idx:
        res = res[:i] + [res[i+1]] + [res[i]] + res[i+2:]
    return "\n".join(res)