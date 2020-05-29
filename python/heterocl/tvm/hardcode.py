def add_loop_label(file_name):
    res = ""
    with open(file_name,"r") as infile:
        loop_cnt = 0
        for line in infile:
            if line[:7] == "    for":
                loop_cnt += 1
                res += "LOOP{}: {}\n".format(loop_cnt,line.strip())
            else:
                res += line
    with open(file_name,"w") as outfile:
        outfile.write(res)