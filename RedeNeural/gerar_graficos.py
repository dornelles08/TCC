import pandas as pd
import matplotlib.pyplot as plt

files = ["se_completo", "se_corolla", "se_onix",
         "sp_completo", "sp_fit", "sp_hb20", "sp_ka"]

for file in files:
    print(file)
    colnames = ['Loss']
    train_losses = pd.read_csv(
        f'./resultados/{file}/train_loss_epoch_{file}.csv', names=colnames, header=None)
    test_losses = pd.read_csv(
        f'./resultados/{file}/test_loss_epoch_{file}.csv', names=colnames, header=None)
    dif_train = pd.read_csv(
        f'./resultados/{file}/train_loss_{file}.csv', names=colnames, header=None)
    dif_test = pd.read_csv(
        f'./resultados/{file}/test_loss_{file}.csv', names=colnames, header=None)

    plt.rc('legend', fontsize=15)
    plt.rcParams['font.size'] = '15'

    plt.figure(figsize=(16, 8))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test', linewidth=3, alpha=0.5)
    plt.xlabel('Épocas', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Convergência', fontsize=20)
    plt.legend(["Treino", "Teste"])
    plt.savefig(
        f'./resultados/{file}/epochs_loss_{file}.png', format='png')

    jump = int(len(dif_train)/200)

    plt.figure(figsize=(16, 8))
    plt.plot(dif_train[::jump], label='Train')
    plt.xlabel('Testes', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Convergence Treino', fontsize=20)
    plt.legend()
    plt.savefig(
        f'./resultados/{file}/car_loss_train_{file}.png', format='png')

    jump = int(len(dif_test)/200)

    plt.figure(figsize=(16, 8))
    plt.plot(dif_test[::jump], label='Test', linewidth=3, alpha=0.5)
    plt.xlabel('Testes', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Convergence Teste', fontsize=20)
    plt.legend()
    plt.savefig(
        f'./resultados/{file}/car_loss_test_{file}.png', format='png')

    plt.close()
