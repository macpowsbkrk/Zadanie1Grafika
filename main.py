import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image_local(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    return img

def show_image(img: np.ndarray, title: str = "Obraz", is_gray: bool = False):
    plt.figure()
    if is_gray:
        plt.imshow(img, cmap="gray")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

def resize_half(img: np.ndarray) -> np.ndarray:
    """Zmniejsza obraz o 50% w obu wymiarach."""
    h, w = img.shape[:2]
    new_w = w // 2
    new_h = h // 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Konwertuje obraz BGR do odcieni szarości."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def rotate_90_clockwise(img: np.ndarray) -> np.ndarray:
    """Obraca obraz o 90 stopni zgodnie z ruchem wskazówek zegara."""
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated

def main():
    path = "images/Yacht.jpg"
    original = load_image_local(path)
    show_image(original, title="Oryginalny obraz")

    resized = resize_half(original)
    show_image(resized, title="Po zmniejszeniu 50%")
def main():
    path = "images/Yacht.jpg"
    original = load_image_local(path)
    show_image(original, title="Oryginalny obraz")

    # 1. Zmniejszenie rozdzielczości
    resized = resize_half(original)
    show_image(resized, title="Po zmniejszeniu 50%")

    # 2. Grayscale
    gray = to_grayscale(resized)
    show_image(gray, title="Po konwersji do szarości", is_gray=True)

    # 3. Obrót o 90°
    rotated = rotate_90_clockwise(gray)
    show_image(rotated, title="Po obrocie o 90°", is_gray=True)

    # 4. Wyświetlenie macierzy
    print("Kształt macierzy:", rotated.shape)
    print("Fragment macierzy (pierwsze 5x5 pikseli):")
    print(rotated[:5, :5])

if __name__ == "__main__":
    main()
