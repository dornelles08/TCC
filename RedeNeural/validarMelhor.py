import json

arq = open("results-Sera?/results.txt", "r")

results = arq.read()
lines = results.split("\n")

print(f"Total de Teste: {len(lines)}")

index, loss = lines[0].split(" ")
best_loss = float(loss)
best_index = index

for line in lines:
    if len(line) != 0:
        index, loss = line.split(" ")
        if best_loss > float(loss):
            best_loss = float(loss)
            best_index = index

print(f"Index Melhor Resultado: {best_index}")
print(f"Melhor Resultado: {best_loss}")

f = open("results-Sera?/result-"+str(index)+".txt", "r")
lines = f.read().split("\n")

args = lines[0]
args = args.replace("{", "")
args = args.replace("}", "")
args = args.split(",")
lr = float(args[2].split(":")[1])
wd = float(args[3].split(":")[1])

print(f"LR: {lr}\nWD: {wd}")
