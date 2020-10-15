import json

with open("adv_log.txt", "r") as f:
    lines = f.readlines()

total = dict()
c = 0
d = 0
e = 0
maxv, maxe = 0, 0
minv, mine = 1e9, 1e9

for line in lines:
    if line.startswith("{"):
        jline = json.loads(line.replace("'", '"'))
        c += 1
        if len(jline) == 1:
            d += 1
            for k, v in jline.items():
                if v == 1:
                    e += 1
        for k, v in jline.items():
            total[k] = total.get(k, 0) + v
totalv = 0
for k, v in total.items():
    totalv += v
print(d, e, len(total) / c, totalv / c)

