import matplotlib.pyplot as plt
import numpy as np
import os

ROTE_BEETE: int = 0

def read_log_file(log_path: str) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """
        Liest Offsets & Metriken aus Datei aus.

        Argumente:
            log_path: str -> Pfad zur Datei

        Gibt zurück:
            tuple[list[int], list[float], list[float], list[float], list[float]] -> Tupel Listen mit Offsets & den Metriken
            (offsets_min, mae, mse, rmse, r2)
    """

    offsets_min: list[int] = []
    mae: list[float] = []
    mse: list[float] = []
    rmse: list[float] = []
    r2: list[float] = []

    with open(log_path, 'r') as file:
        for line in file:
            # Leerzeilen überspringen
            if line == '\n':
                continue

            split_line: list[str] = line.split()

            if split_line[0].startswith("offset="):
                offsets_min.append(int(split_line[2]))
                continue

            match split_line[0]:
                case "mae:":
                    mae.append(float(split_line[1]))
                case "mse:":
                    mse.append(float(split_line[1]))
                case "rmse:":
                    rmse.append(float(split_line[1]))
                case "r2:":
                    r2.append(float(split_line[1]))
                case _:
                    print(f"Unerwartete Zeile gefunden:\n{line}")
    
    return (offsets_min, mae, mse, rmse, r2)

def plot_metrics(offsets_min: list[int], mae: list[float], mse: list[float], rmse: list[float], r2: list[float]) -> None:
    """
        Plottet Metriken in Abhängigkeit des Vorhersagenoffsets.

        Erstellt Diagramm mit vier Subplots, speichert es ab und öffnet es falsch gewünscht.

        Argumente:
            offsets_min: list[int] -> Vorhersagenoffsets in Minuten
            mae: list[float] -> Listen mit den jeweiligen Metriken
            mse: list[float] -> .
            rmse: list[float] -> .
            r2: list[float] -> .
    """
    plt.subplot(2, 2, 1)
    plt.plot(offsets_min, mae, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Mean Absolute Error")
    plt.grid(which="major", axis="both", color="#B4B4B4", linestyle="dotted", linewidth=.2)
    # y-Achse
    plt.ylabel("MAE")
    # x-Achse
    plt.xlabel("Offset in Minuten")
    plt.xticks(np.arange(0, max(offsets_min)+1, 60), minor=False) # Zahlen an x-Achse
    plt.xticks(np.arange(0, max(offsets_min)+1, 10), minor=True) # Zwischenschritte zwischen Zahlen

    plt.subplot(2, 2, 2)
    plt.plot(offsets_min, mse, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Mean Squared Error")
    plt.grid(which="major", axis="both", color="#B4B4B4", linestyle="dotted", linewidth=.2)
    # y-Achse
    plt.ylabel("MSE")
    # x-Achse
    plt.xlabel("Offset in Minuten")
    plt.xticks(np.arange(0, max(offsets_min)+1, 60), minor=False) # Zahlen an x-Achse
    plt.xticks(np.arange(0, max(offsets_min)+1, 10), minor=True) # Zwischenschritte zwischen Zahlen

    plt.subplot(2, 2, 3)
    plt.plot(offsets_min, rmse, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("Root Mean Squared Error")
    plt.grid(which="major", axis="both", color="#B4B4B4", linestyle="dotted", linewidth=.2)
    # y-Achse
    plt.ylabel("RMSE")
    # x-Achse
    plt.xlabel("Offset in Minuten")
    plt.xticks(np.arange(0, max(offsets_min)+1, 60), minor=False) # Zahlen an x-Achse
    plt.xticks(np.arange(0, max(offsets_min)+1, 10), minor=True) # Zwischenschritte zwischen Zahlen

    plt.subplot(2, 2, 4)
    plt.plot(offsets_min, r2, linestyle='', marker='.', ms=3, mec="#4CAF50", mfc="#4CAF50")
    plt.title("R2 Score")
    plt.grid(which="major", axis="both", color="#B4B4B4", linestyle="dotted", linewidth=.2)
    # y-Achse
    plt.ylabel("R2")
    # x-Achse
    plt.xlabel("Offset in Minuten")
    plt.xticks(np.arange(0, max(offsets_min)+1, 60), minor=False) # Zahlen an x-Achse
    plt.xticks(np.arange(0, max(offsets_min)+1, 10), minor=True) # Zwischenschritte zwischen Zahlen

    plt.suptitle("Metriken in Abhängigkeit der Vorhersagenoffsets", size=16)
    plt.tight_layout()

    save_path = input("Wo soll das Diagramm gespeichert werden? ")
    plt.savefig(save_path)

    open_fig = input("Diagramm anzeigen? (y/n) ")
    if open_fig == 'y':
        os.system(f"xdg-open {save_path}")

"""
    Auswertung des Tests verschiedener Vorhersagetiefen.

    Liest .log-Datei ein und erstellt ein Diagramm der verschiedenen Metriken in Abhängigkeit der Vorhersagetiefe (Offset zwischen letztem X- und erstem y-Wert). 
"""
if __name__ == "__main__":
    PROJECT_DIR: str = os.path.join(os.path.dirname(__file__), "..")

    # Pfad zur .log-Datei, relativer Pfad vom Projektverzeichnis (ai4all-projekt/) aus
    log_path: str = input("Pfad zur .log-Datei (relativ zum Projektverzeichnis): ")
    # log_path = "logs/vorhersagentiefe.log"

    # Test ob Datei existiert
    if not os.path.isfile(os.path.join(PROJECT_DIR, log_path)):
        print("ERROR: Die angegebene Datei existiert nicht!")
        exit(1)

    offsets_min: list[int] = []
    mae: list[float] = []
    mse: list[float] = []
    rmse: list[float] = []
    r2: list[float] = []

    print(f"Daten aus {log_path} auslesen..")
    offsets_min, mae, mse, rmse, r2 = read_log_file(os.path.join(PROJECT_DIR, log_path))

    print("Metriken plotten..")
    plot_metrics(offsets_min, mae, mse, rmse, r2)