import json

arq = open("results-1/results.txt", "r")

results = arq.read()
lines = results.split("\n")

print(len(lines))

bests_index = []
bests_loss = []

for line in lines:
  if len(line) != 0:
    index, loss = line.split(" ")
    if 2000 > float(loss):
      bests_loss.append(float(loss))
      bests_index.append(index)

print(f"Qtd: {len(bests_index)}")
print(f"%: {(len(bests_index)/len(lines))*100}")

lrs = []
wds = []
for index in bests_index:
  f = open("results-1/result-"+str(index)+".txt", "r")
  lines=f.read().split("\n")

  args = lines[0]
  args = args.replace("{", "")
  args = args.replace("}", "")
  args = args.split(",")
  lr = float(args[2].split(":")[1])
  wd = float(args[3].split(":")[1])
  if lr not in lrs:
    lrs.append(lr)
  if wd not in wds:
    wds.append(wd)

# print(lrs, wds)