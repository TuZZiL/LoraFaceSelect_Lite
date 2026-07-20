# Project Report

## 2026-07-19

- context: Перевірка сумісності локальної ONNX-моделі з ONNX Runtime та OpenCV DNN.
- done: NudeNet SHA-256 збігається із зафіксованим upstream hash; `onnx.checker(full_check=True)` успішний.
- done: ONNX Runtime 1.23.1 виконує NCHW `(1,3,320,320)` і повертає `(1,22,2100)`.
- done: OpenCV DNN 4.12.0 відтворено падає на `/model.11/Concat` через dynamic shapes.
- context: Після static input/shape inference OpenCV проходить `Concat`, але падає далі на `/model.22/ConstantOfShape`.
- resolved: Модель валідна, але поточний OpenCV DNN backend несумісний із графом не лише через output metadata.
- context: NudeNet опціональний; pipeline продовжує працювати з body landmarks без NudeNet.
- done: NudeNet переведено на окремий ONNX Runtime CPU backend; решта моделей лишилась на OpenCV DNN.
- done: `onnxruntime>=1.23.1` додано в базові dependencies для однакової роботи CLI та GUI.
- done: Тести адаптовано під ORT session API; повний suite — 49 passed.
- done: `doctor` і GUI CLI-route показують `NudeNet-v3-320n/ONNXRuntime-CPU`; inference/decode/NMS перевірено на 5 реальних фото.
- resolved: OpenCV import error NudeNet усунуто без re-export моделі та без зміни решти pipeline.
- done: README доповнено Windows PowerShell quick start для GUI через окрему `.venv-win`, `doctor` і порядок дій у вікні.
- done: Проект очищено від Linux setup: видалено generated `.venv` і `.venv-mobileclip` (~1.33 GB), README переведено на PowerShell, ignore/tool docs — на Windows venv names.
- resolved: Довгий GUI-прогін на 1710 фото з OpenCV 5 активно рахував, але RAM виросла приблизно до 13 GB, а progress не відображався коректно.
- done: OpenCV зафіксовано на `>=4.10,<5`; `.venv-win` використовує 4.13.0, warning нового graph engine зник.
- done: Large-image decode оптимізовано через Pillow `draft`/`thumbnail` до RGB conversion.
- done: MobileCLIP/body/NudeNet пропускаються для hard-rejected кадрів; fallback-eligible поведінку збережено.
- done: GUI redirect тепер показує carriage-return progress як `current/total`; додано regression test.
- done: Stress test 180 images: RAM 319→395 MB без лінійного росту; stable/experimental doctor успішні; 51 test passed.
- done: Старий GUI process tree завершено після перевірки fix; OpenCV 5 uninstall residue 81.9 MB видалено.
- done: Групові фото з чітким target тепер беруть участь у відборі; `prepared/` геометрично виключає всі інші detected face bbox навіть після bucket expansion.
- resolved: Кадри з перекритими/надто близькими face bbox лишаються у `multiple_faces_review`; повний suite — 56 passed.
- done: Додано кореневий `run_gui.bat` для запуску GUI через `.venv-win` із перевіркою environment та видимою помилкою запуску.
- resolved: `prepared/` більше не зберігає довільні aspect ratios при `no_compatible_bucket`; context безпечно trim-иться до точного Krea bucket, замалі кадри пропускаються без апскейлу.
- done: Multi-face far кадри готуються як target-centric portrait; сторонні tight face bbox розширено до head-zone, конфліктні кадри йдуть у review.
- done: Selection резервує до 30% strong набору під single-person far/body кадри; manifest/review містять `training_focus=body|identity`.
- done: Реальні проблемні `005`/`020` перевірено як чисті `512x768`; `018` переведено в review через близькі head-zones; повний suite — 62 passed.
- context: У `resultIlNEw` вибрано 30, підготовлено 27; ranks 023/024/025 — multi-face (2/5/2 faces), для яких немає стандартного Krea crop без сторонніх head-zones.
- done: GUI після base crop показує modal із діями backfill, tighter identity crop або finish; callback працює в поточному run без повторного face analysis.
- done: Невдалі crops копіюються у `crop_skipped/` з `crop_skips.csv`; `selected/` синхронізується лише з готовими `prepared/` slots.
- done: Backfill перебирає наступні ranked candidates до requested count або вичерпання; tighter crop не запускає body inference і не послаблює head safety.
- done: Summary містить base/prepared/resolved/skipped counts, action і session status; real smoke backfill/tight успішний, 65 tests passed.
- context: Користувач сам завантажує локальні відео; Instagram download/automation не входить у scope.
- done: Додано послідовний video sampler із default 0.5 fps, max 120 samples і max 3 diverse target-face candidates на відео; RAM обмежена top-K одного файла.
- done: `video_progress.json` checkpoint зберігається після кожного відео; cache invalidation враховує файл, reference identity, backend і video settings.
- done: Відеокадри проходять чинний global identity/quality/diversity/body/crop pipeline; report і dataset manifest містять source video, timestamp та frame number.
- done: GUI має compact Local videos controls і явний Resume/refresh checkbox; CLI підтримує `--no-analyze-videos` та video limits.
- resolved: Real MJPG/OpenCV VideoCapture smoke — 5 sampled, 3 kept; seek/decode/JPEG/checkpoint path працює.
- done: Frontend critique виправив disabled video controls, primary Run action, busy button states і vertical layout weight.
- done: Simplify/verification завершено; atomic frame writes, compileall clean, повний suite — 70 passed.
- context: Діагностика `Bkl4VID1`: video ranks 007/014 містять чоловіка, але YuNet повернув `face_count=1`; профільне/нахилене стороннє обличчя не потрапило в safety geometry.
- context: 007/014 пройшли identity/quality gates (`sim` 0.565/0.576, `quality` 0.541/0.583) і diversity ranking; поточний ranking не штрафує стороннє тіло без detected face.
- ideas: Потрібні stronger secondary-face/person guard для video frames і явний результат modal action (`0 resolved`) замість враження, що кнопка не спрацювала.

## 2026-07-20

- context: Підготовка проекту до публікації на GitHub та створення презентаційної сторінки.
- done: Перевірено працездатність pytest suite (70 passed).
- done: Оновлено `.gitignore` для очищення репозиторію від тимчасових тестових каталогів та кешу.
- done: Створено файл `LICENSE` (MIT License).
- done: Створено веб-сторінку GitHub Pages (`docs/index.html`) з інтерактивним оглядом пайплайну, калькулятором Krea-бакетів та гайдом.
- done: Модернізовано `README.md` з бейджами, архітектурними діаграмами та вичерпною документацією.
- done: Ініціалізовано локальний git, додано origin remote, здійснено коміт та перший `git push` до GitHub репозиторію.
- done: Додано механізм відкидання викидів (Outliers) та якісно-зваженого середнього референсних векторів.
- done: Реалізовано адаптивний поріг схожості для профілів (Pose-Adaptive Threshold).
- done: Додано 3 нових юніт-тести в `tests/test_analysis.py`, повний тестовий пакет успішно пройдено (73 passed).
- done: Дозволено кроп тіла (body-aware crop) для зображень далекого ракурсу (`scale_bin == "far"`), якщо знайдено контури тіла, замість постійного збереження повного кадру.
- done: Додано юніт-тест для перевірки `body_crop_far` стратегії, всі 74 тести успішно пройшли.
- done: Завантажено та верифіковано опціональну сумісну модель MobileCLIP-S0 ONNX з репозиторію `plhery/mobileclip2-onnx` на Hugging Face (попередня S2 модель падала в OpenCV DNN на кроці shape inference).


