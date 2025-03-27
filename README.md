# Diver detection



## Commands:

run yolo gui:
```bash
yolo predict model=yolo11n.pt show=True
```

## Download everything
```bash
python download_vddc.py --all
```

## Download just what we need for YOLO training
```bash
python download_vddc.py --images --yolo-labels
```

## Download to a custom location
```bash
python download_vddc.py --images --yolo-labels --output-dir /custom/path
```

## refs:

https://docs.ultralytics.com/guides/nvidia-jetson/#quick-start-with-docker