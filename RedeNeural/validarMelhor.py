arq = open("results/results.txt", "r")

results = arq.read()
lines = results.split("\n")

print(len(lines))

index, loss = lines[0].split(" ")
best_index = int(index)
best_loss = float(loss)

for line in lines:
  if len(line) != 0:
    index, loss = line.split(" ")
    if float(best_loss) > float(loss):
      best_loss = float(loss)
      best_index = int(index)

print(best_index, " ", best_loss)

f = open("results/result_"+str(best_index)+".txt", "r")
lines=f.read().split("\n")
print(lines[0])