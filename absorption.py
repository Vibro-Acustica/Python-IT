import numpy as np


def calculate_absorption_coeficent(self, freq, H, l, s, c=343):
        """
        Calcula o coeficiente de absorção acústica a partir de H (complexo).

        Parâmetros:
        - freq: array de frequências (Hz)
        - H: FRF complexa
        - l: distância da amostra até o centro do microfone mais próximo (em metros)
        - s: distância entre os centros dos microfones (em metros)
        - c: velocidade do som (m/s), padrão: 343 m/s

        Retorna:
        - alpha: array com o coeficiente de absorção
        """
        freq = np.asarray(freq)
        H = np.asarray(H)
        
        Hr = H.real
        Hi = H.imag
        Hr2 = Hr**2
        Hi2 = Hi**2

        k = 2 * np.pi * freq / c
        Hr2 = Hr**2
        Hi2 = Hi**2

        D = 1 + Hr2 + Hi2 - 2 * (Hr * np.cos(k * s) + Hi * np.sin(k * s))

        Rr = (2 * Hr * np.cos(k * (2 * l + s)) - np.cos(2 * k * l) - (Hr2 + Hi2) * np.cos(2 * k * (l + s))) / D
        Ri = (2 * Hr * np.sin(k * (2 * l + s)) - np.sin(2 * k * l) - (Hr2 + Hi2) * np.sin(2 * k * (l + s))) / D

        alpha = 1 - Rr**2 - Ri**2
        
        # Ensure values are within physical limits (sometimes numerical issues can cause values >1)
        alpha = np.clip(alpha, 0, 1)
        
        return alpha