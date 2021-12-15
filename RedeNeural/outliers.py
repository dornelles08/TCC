import psycopg2

conn = psycopg2.connect(
    host="10.0.0.185",
    database="TCC",
    user="postgres",
    password="123456")

cur = conn.cursor()

file = open("valorFipe.txt", 'r')
# carro5 = open("carro5.txt", 'w')
# carro10 = open("carro10.txt", 'w')
# carro15 = open("carro15.txt", 'w')
# carro20 = open("carro20.txt", 'w')

lines = file.read().split('\n')

lines = lines[1:]


def salvaCarro(percent, carro):
    data = f"{carro[0]},{carro[1]},{carro[2]},{carro[3]},{carro[4]},{carro[5]},{carro[6]},{carro[7]},{carro[8]},{carro[9]},{carro[10]},{carro[11]},{carro[12]},{carro[13]},{carro[14]},{carro[15]},{carro[16]},{carro[17]},{carro[18]},{carro[19]},{carro[20]},{carro[21]},{carro[22]}\n"
    dataSQL = f"'{carro[0]}','{carro[1]}','{carro[2]}','{carro[3]}','{carro[4]}','{carro[5]}','{carro[6]}','{carro[7]}','{carro[8]}','{carro[9]}','{carro[10]}','{carro[11]}','{carro[12]}','{carro[13]}','{carro[14]}','{carro[15]}','{carro[16]}','{carro[17]}','{carro[18]}','{carro[19]}','{carro[20]}','{carro[21]}','{carro[22]}'\n"

    if percent <= 5:
        # carro5.write(data)
        cur.execute(f"INSERT INTO carros5 VALUES ({dataSQL})")
    elif percent > 5 and percent <= 10:
        # carro10.write(data)
        cur.execute(f"INSERT INTO carros10 VALUES ({dataSQL})")
    elif percent > 10 and percent <= 15:
        # carro15.write(data)
        cur.execute(f"INSERT INTO carros15 VALUES ({dataSQL})")
    elif percent > 15 and percent <= 20:
        # carro20.write(data)
        cur.execute(f"INSERT INTO carros20 VALUES ({dataSQL})")


for line in lines:
    try:
        carro = line.split(";")
        marca = carro[0]
        modelo = carro[1]
        ano = carro[2]
        valor = carro[3]

        cur.execute(
            f"SELECT * FROM modelo WHERE description = '{marca} {modelo}'")

        if cur.rowcount > 0:
            cur.execute(
                f"SELECT * FROM carros WHERE modelo = '{marca} {modelo}' and ano = {ano}")
            for c in cur.fetchall():
                carValue = float(c[22])
                fipeValue = float(valor)
                if carValue > 0:
                    if carValue < fipeValue:
                        percent = int((fipeValue*100)/carValue)
                    else:
                        percent = int((carValue*100)/fipeValue)-100
                    salvaCarro(percent, c)
    except IndexError:
        print(line)
