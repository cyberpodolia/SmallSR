# smallSR — Lightweight Super-Resolution & Denoising Pipeline

`smallSR` — це компактний, оптимізований пайплайн суперрезолюшну (SR) та денойзу, розроблений спеціально для роботи з низькороздільними потоками камер (наприклад PS3 Eye) та для подальшого розгортання на малопотужних пристроях типу **Jetson Nano**.

Проєкт включає:
- мультимодельний «teacher» енхансер (NAFNet, RealESRGAN, інші),
- невелику «student» модель (`student_sr.pth`), навчення якої відбувається на результатах teacher-пайплайну,
- Python-скрипти для тренування, інференсу, конвертації моделей в ONNX/FP16,
- готовий image-pipeline для пакетної обробки.

Проєкт структурований максимально просто, щоб можна було легко повторити, адаптувати або оптимізувати під GPU/RTX або під ARM-платформи.

---

# 🚀 Основні можливості

- **Суперрезолюшн 4×** для низькороздільних камер (640×480 → 1280×1024).
- **Денойз** + частковий **деблюр** (teacher-пайплайн).
- **Student-модель**, яка повторює роботу складних енханcерів, але працює у реальному часі.
- **Сумісність з ONNX та TensorRT**.
- **Пакетна обробка директорій** через `image_pipeline`.
- **Режим реального часу**, включно з PS3 Eye камерою.

---

# 📂 Структура проєкту

mallSR/
│
├── python/
│ ├── capture_ps3_camera.py # Захоплення кадрів PS3 Eye
│ ├── RUN_STUDENT_TEST.py # Інференс student-моделі
│ ├── image_pipeline/ # Моделі, пайплайни, обробка зображень
│ ├── train/ # Тренування student-моделі
│ └── ... # Утиліти, скрипти конвертації, тестування
│
├── weights/
│ └── student_sr.pth # Єдина вага, яку трекає Git
│
├── .gitignore
├── README.md
├── requirements.txt
└── project_manifest.txt


---

# 📸 Teacher → Student концепція

Teacher-пайплайн складається з сильних моделей (NAFNet, RealESRGAN, інші), які створюють high-quality ground truth.

Потім student-модель навчається **імітувати результат teacher-пайплайна**, але у значно компактнішій архітектурі.

Переваги:
- можна запускати на слабких GPU,
- inference в реальному часі,
- навчання відбувається без ручної розмітки.

---

# 🔧 Інсталяція


git clone https://github.com/Yondr/smallSR.git
cd smallSR
pip install -r requirements.txt



▶️ Запуск студентської моделі
python python/RUN_STUDENT_TEST.py --input path/to/input --output out/


Або завантаж зображення у python/get_data_from_cam.py щоб знімати з PS3 камери.

---

# 🚀 Real-time Inference

To run the real-time super-resolution pipeline with a PS3 Eye camera, first make sure you have the required packages installed from `requirements.txt`. It is recommended to use a virtual environment.

Then, use the following command from the `project_clean` directory:

```bash
python python/realtime_trt_student.py
```

This will open two windows:
-   **Original 640x480**: The raw input from the PS3 Eye camera.
-   **Student SR TensorRT (1280x960)**: The super-resolution output from the TensorRT model.

Press `ESC` to close the windows and stop the script.

---

# 🧪 Тренування student-моделі

train.train_student_sr

python -m train.train_student_sr `
>>   --raw-dir "Y:\gemini\project_clean\data\ps3_captures_vol2" `
>>   --hires-dir "Y:\gemini\project_clean\data\ps3_captures_processed_vol2" `
>>   --raw-val-dir "Y:\gemini\project_clean\data\validation_lr" `
>>   --hires-val-dir "Y:\gemini\project_clean\data\validation_hr" `
>>   --out "Y:\gemini\project_clean\weights\student_sr.pth" `
>>   --epochs 20 --batch-size 8 --patch-size 128 --num-workers 4    

🧱 Конвертація в ONNX
python python/convert_to_onnx.py --weights weights/student_sr.pth

🎯 Ціль проєкту

Створити мінімальний, ефективний, легкий SR+денойз пайплайн, який:
працює на звичайних відеопотоках (PS3 Eye, USB webcams),
забезпечує високу якість через teacher→student pipeline,
може бути розгорнутий на Jetson Nano в режимі реального часу.

---

# 📊 Логи тренування

Директорія `training_logs/` містить результати запусків тренування student-моделі.

- `training_log.csv`: CSV-файл з метриками loss та PSNR для кожної епохи.
- `loss_plot.png`: Графік, що показує тренувальний та валідаційний loss по епохам.
- `psnr_plot.png`: Графік, що показує тренувальний та валідаційний PSNR по епохам.

Ці файли генеруються скриптами `train_student_sr.py` та `plot_metrics.py`.
