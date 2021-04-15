
file = open("Logs/COCO training log 5 epochs 0.001LR.txt", "r")
loss = 0
list = []
for line in file:
        list.append(line)

for each in list:
    x = each.split()[5]
    x = x.split(",")
    loss = loss + float(x[0])

loss = loss/5800
print(loss)