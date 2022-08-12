import pandas as pd
import matplotlib.pyplot as plt

files = ["se_completo", "se_corolla", "se_onix",
         "sp_completo", "sp_fit", "sp_hb20", "sp_ka"]

for file in files:
    print("=============================================")
    print(file)
    colnames = ['Loss']
    dif_train = pd.read_csv(
        f'./resultados/{file}/train_loss_{file}.csv', names=colnames, header=None)
    dif_test = pd.read_csv(
        f'./resultados/{file}/test_loss_{file}.csv', names=colnames, header=None)

    dif_array = []

    skip = 100
    start = 0
    end = skip

    for i in range(int(len(dif_train)/100)):
        dif_array.append(round(dif_train[start:end].mean().Loss, 2))
        start = end
        end = start + skip
        if(end > len(dif_train)):
            end = len(dif_train)

    plt.rc('legend', fontsize=15)
    plt.rcParams['font.size'] = '15'

    jump = int(len(dif_array)/200)

    plt.figure(figsize=(16, 8))
    plt.plot(dif_array[::jump], label='Train')
    plt.xlabel('Testes', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Convergence Treino', fontsize=20)
    plt.legend()
    plt.savefig(
        f'./resultados/{file}/car_loss_train_{file}.png', format='png')

    dif_array = pd.DataFrame(dif_array)

    print("Treino")
    print(dif_array.min())
    print(dif_array.mean())
    print(dif_array.max())
    print("______________________________________________________")

    dif_array = []

    skip = 100
    start = 0
    end = skip

    for i in range(int(len(dif_test)/100)):
        dif_array.append(round(dif_test[start:end].mean().Loss, 2))
        start = end
        end = start + skip
        if(end > len(dif_test)):
            end = len(dif_test)

    jump = int(len(dif_array)/200)

    plt.figure(figsize=(16, 8))
    plt.plot(dif_array[::jump], label='Test', linewidth=3, alpha=0.5)
    plt.xlabel('Testes', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Convergence Teste', fontsize=20)
    plt.legend()
    plt.savefig(
        f'./resultados/{file}/car_loss_test_{file}.png', format='png')

    dif_array = pd.DataFrame(dif_array)

    print("Teste")
    print(dif_array.min())
    print(dif_array.mean())
    print(dif_array.max())
    print("______________________________________________________")

    plt.close()
