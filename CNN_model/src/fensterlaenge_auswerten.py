import matplotlib.pyplot as plt
import numpy as np
import os
import sys

"""
    Auswertung eines abgebrochenen Versuchs zur Fensterlänge.

    Liest Daten des abgebrochenen Versuchs aus data/metrics_partial.npz ein und visualisiert diese.
"""

def load_and_append(path: str, mae_list: list[float], mse_list: list[float], rmse_list: list[float], r2_list: list[float]) -> tuple[list[float],list[float],list[float],list[float]]:
    data = np.load(path)

    mae_list.append(data["mae"])
    mse_list.append(data["mse"])
    rmse_list.append(data["rmse"])
    r2_list.append(data["r2"])

    return (mae_list, mse_list, rmse_list, r2_list)

if __name__ == "__main__":
    mae_list: list[float] = []
    mse_list: list[float] = []
    rmse_list: list[float] = []
    r2_list: list[float] = []

    PROJECT_DIR: str = os.path.join(os.path.dirname(__file__), "..")
    
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/metrics_partial_8_20.npz")
    mae_list, mse_list, rmse_list, r2_list = load_and_append(DATA_PATH, mae_list, mse_list, rmse_list, r2_list)
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/metrics_partial_20_30.npz")
    mae_list, mse_list, rmse_list, r2_list = load_and_append(DATA_PATH, mae_list, mse_list, rmse_list, r2_list)
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/metrics_partial_30_40.npz")
    mae_list, mse_list, rmse_list, r2_list = load_and_append(DATA_PATH, mae_list, mse_list, rmse_list, r2_list)
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/metrics_partial_40_50.npz")
    mae_list, mse_list, rmse_list, r2_list = load_and_append(DATA_PATH, mae_list, mse_list, rmse_list, r2_list)
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/metrics_partial_50_60.npz")
    mae_list, mse_list, rmse_list, r2_list = load_and_append(DATA_PATH, mae_list, mse_list, rmse_list, r2_list)
    DATA_PATH: str = os.path.join(PROJECT_DIR, "data/metrics_partial_60_65.npz")
    mae_list, mse_list, rmse_list, r2_list = load_and_append(DATA_PATH, mae_list, mse_list, rmse_list, r2_list)

    mae_arr = np.concatenate(mae_list, axis=0)
    mse_arr = np.concatenate(mse_list, axis=0)
    rmse_arr = np.concatenate(rmse_list, axis=0)
    r2_arr = np.concatenate(r2_list, axis=0)

    x = np.arange(8, 8 + len(mae_arr))

    a, b = np.polyfit(x, mae_arr, 1)
    plt.subplot(2, 2, 1)
    plt.plot(x, a*x+b, linestyle='-', color="#FF484B")
    plt.plot(x, mae_arr, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Mean Absolute Error")
    plt.xlabel("Fensterlänge")
    plt.ylabel("MAE")

    a, b = np.polyfit(x, mse_arr, 1)
    plt.subplot(2, 2, 2)
    plt.plot(x, a*x+b, linestyle='-', color="#FF484B")
    plt.plot(x, mse_arr, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Mean Squared Error")
    plt.xlabel("Fensterlänge")
    plt.ylabel("MSE")

    a, b = np.polyfit(x, rmse_arr, 1)
    plt.subplot(2, 2, 3)
    plt.plot(x, a*x+b, linestyle='-', color="#FF484B")
    plt.plot(x, rmse_arr, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Root Mean Squared Error")
    plt.xlabel("Fensterlänge")
    plt.ylabel("RMSE")

    a, b = np.polyfit(x, r2_arr, 1)
    plt.subplot(2, 2, 4)
    plt.plot(x, a*x+b, linestyle='-', color="#FF484B")
    plt.plot(x, r2_arr, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("R2 Score")
    plt.xlabel("Fensterlänge")
    plt.ylabel("R2")

    plt.tight_layout()

    save_path: str = input("Wo soll das Diagramm gespeichert werden? ")
    print(f"Diagramm wird unter {save_path} gespeichert.")
    plt.savefig(save_path)

    show_plot: str = input("\nDiagramm öffnen? (y/n) ")
    if show_plot == 'y':
        os.system(f"xdg-open {save_path}")