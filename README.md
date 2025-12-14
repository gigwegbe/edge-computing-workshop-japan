# Edge Computing Workshop Japan 2025

![Cover Image](./asset/header-image.jpg)

- Edge Computing Slide - [Link](https://docs.google.com/presentation/d/17RwHiok1fHg5p3La4Y7FH_nfN_jkB-7isJlzI7WQRE8/edit?usp=sharing)

### Prerequisites

- Ensure you're using Python 3.11 - 3.13.
- [uv](https://docs.astral.sh/uv/) package manager or [pip](https://pypi.org/project/pip/)

```bash

- Create a python virtual environment

```

python -m venv edge-env

```


- Change the directory to `image-classification`
```

cd image-classification

```
- Inference on live video stream from camera (Unoptimized Model - float 32): 
```

(edge-env)python3 camera_infer_h5.py -m models/model.h5 -l labels.txt --width 320 --height 320

```
- Inference on live video stream from camera (Optimized Model - int-8): 
```

python camera_infer_tflite.py --model models/ei-edge-computing-workshop-2025-image-classification-classifier-tensorflow-lite-int8-quantized-model.3.lite --labels labels.txt --camera 0 --top_k 3

```

Quit the live stream by pressing `q` or `ctrl + c`: 
```

^CTraceback (most recent call last):
  File "/Users/george/Documents/github/edge-computing-workshop-kigali/image-classification/camera_infer_tflite.py", line 162, in <module>
    main()
  File "/Users/george/Documents/github/edge-computing-workshop-kigali/image-classification/camera_infer_tflite.py", line 154, in main
    key = cv2.waitKey(1) & 0xFF
          ^^^^^^^^^^^^^^
KeyboardInterrupt
^C

```

- Inference on images saved on disk: 
```

python batch_infer_images.py --model models/ei-edge-computing-workshop-2025-image-classification-classifier-tensorflow-lite-int8-quantized-model.3.lite --labels labels.txt --images_dir ./sample-directory

```
