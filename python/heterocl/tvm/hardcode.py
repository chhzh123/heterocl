def hardcode(filename):
    code = open(filename,"r").readlines()
    new_code = move_const(code)
    with open(filename,"w") as outfile:
        outfile.write(new_code)
    print("Done hardcode!")

def move_const(code):
    const = []
    res = []
    for line in code:
        if "const" in line:
            const.append(line.strip()+"\n")
        else:
            res.append(line)
    res = res[:11] + ["\n"] + const + ["\n"] + res[11:]
    return "".join(res)