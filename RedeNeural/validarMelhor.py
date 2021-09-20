import json

arq = open("results/results.txt", "r")

results = arq.read()
lines = results.split("\n")

print(len(lines))

bests_index = []
bests_loss = []

for line in lines:
  if len(line) != 0:
    index, loss = line.split(" ")
    if 34000 > float(loss):
      bests_loss.append(float(loss))
      bests_index.append(index)

print(bests_index)
print(bests_loss)

print('\n\n')
lrs = []
wds = []
for index in bests_index:
  f = open("results/result_"+str(index)+".txt", "r")
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

print(lrs, wds)