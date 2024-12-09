import cv2
import mediapipe as mp
import os
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

# Инициализация объектов для обработки лиц
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Папка с изображениями
image_folder = 'input_images'
output_folder = 'output_images'
style_images = 'style_images'

# Проверяем, существует ли папка для сохранения изображений, если нет - создаем её
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Загружаем модель для переноса стиля
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Используем обновленный способ загрузки предобученной модели для VGG19
from torchvision.models import VGG19_Weights
style_model = models.vgg19(weights=VGG19_Weights.DEFAULT).to(device).eval()

# Функция для улучшения качества изображения с использованием OpenCV
def enhance_image_opencv(image):
    # Увеличиваем изображение с помощью интерполяции (например, INTER_CUBIC)
    height, width = image.shape[:2]
    new_size = (width * 4, height * 4)  # Увеличиваем изображение в 4 раза
    enhanced_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)  # Или cv2.INTER_LANCZOS4
    return enhanced_image

# Преобразование изображения для стилизации
def image_to_tensor(image_path, target_size=(1024, 1024)):
    image = Image.open(image_path).convert("RGB")  # Преобразуем в RGB, если это необходимо
    preprocess = transforms.Compose([
        transforms.Resize(target_size),  # Меняем на нужный размер, например, 1024x1024
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0).to(device)

# Применение стиля с использованием нейронной сети
def apply_style(content_image_path, style_image_path, style_weight=1.0, content_weight=1.0):
    content_image = image_to_tensor(content_image_path)
    style_image = image_to_tensor(style_image_path)

    # Получаем изображения с учетом весов
    styled_image = style_weight * style_image + content_weight * content_image

    # Преобразуем результат в правильный формат
    styled_image = styled_image.squeeze(0).cpu().detach().numpy()  # Убираем batch dimension
    styled_image = np.transpose(styled_image, (1, 2, 0))  # Меняем порядок осей на (H, W, C)

    # Нормализуем значения пикселей в диапазоне [0, 255]
    styled_image = np.clip(styled_image * 255, 0, 255).astype('uint8')
    return styled_image

# Функция для обработки изображений и проверки наличия лиц
def detect_faces_in_image(image_path, style_weight=1.0, content_weight=1.0):
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка при загрузке изображения: {image_path}")
        return

    # Преобразуем изображение в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Детекция лиц
    with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
        results = face_detection.process(image_rgb)

    # Если лицо найдено, рисуем его на изображении
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

        # Улучшаем качество изображения перед применением стиля
        enhanced_image = enhance_image_opencv(image)  # Используем улучшение изображения с OpenCV

        # Указываем путь к файлу стиля
        style_image_path = os.path.join(style_images, '1.jpg')

        if not os.path.exists(style_image_path):
            print(f"Ошибка: файл стиля не найден: {style_image_path}")
        else:
            # Применяем стиль только к изображению с лицом с учетом веса
            styled_image = apply_style(image_path, style_image_path, style_weight, content_weight)

            # Преобразуем в BGR для OpenCV
            styled_image_bgr = cv2.cvtColor(styled_image, cv2.COLOR_RGB2BGR)

            # Сохраняем стилизованное изображение в папку output_images
            output_image_path = os.path.join(output_folder, os.path.basename(image_path))
            cv2.imwrite(output_image_path, styled_image_bgr)
            print(f"{image_path} - Стилизованное изображение сохранено в {output_image_path}")
    else:
        print(f"{image_path} - Лицо не распознано")


# Перебираем все файлы в папке input_images
for filename in os.listdir(image_folder):
    # Формируем полный путь к изображению
    image_path = os.path.join(image_folder, filename)

    # Проверяем, является ли файл изображением (по расширению)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Задаем веса для стиля и контента
        style_weight = 1.3  # вес стиля
        content_weight = 2  # вес контента

        # Обрабатываем изображение с заданными весами
        detect_faces_in_image(image_path, style_weight, content_weight)

# Закрываем все окна OpenCV
cv2.waitKey(0)
cv2.destroyAllWindows()
