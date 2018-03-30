import matplotlib.pyplot as plt

def plot_losses(train_hist):
    d_loss = train_hist["D_loss"]
    g_loss = train_hist["G_loss"]

    x = list(range(len(d_loss)))
    plt.plot(x, d_loss, label="Discriminator Loss")
    plt.plot(x, g_loss, label="Generator Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    plt.show()
