with open("clean_train.txt", "w") as fout:
    with open("train.txt", "r") as fin:
        for line in fin:
            line_ = line.strip()
            if line_.split("</UTT>")[-1] == "1":
                print(line_[:-1], file=fout)

with open("clean_valid.txt", "w") as fout:
    with open("valid.txt", "r") as fin:
        for line in fin:
            line_ = line.strip()
            if line_.split("</UTT>")[-1] == "1":
                print(line_[:-1], file=fout)