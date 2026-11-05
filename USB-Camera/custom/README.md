# 7종 물체 검출기

## 클래스

| 키 | 이름 | 대상 |
|---|---|---|
| 1 | `aa_battery` | AA 건전지 |
| 2 | `box` | 정상 원목 큐브 (휴지 그림이 붙어 있음) |
| 3 | `eraser` | 지우개 |
| 4 | `vita500_capped` | 뚜껑 있고 라벨이 온전한 비타500 빈 병 |
| 5 | `ping_pong_ball` | 정상 탁구공 |
| 6 | `mini_paper_cup` | 정상 미니 종이컵 |
| 7 | `defective` | 불량품: 뚜껑 없거나 라벨이 손상된 비타500, 찌그러진 탁구공, 찌그러진 종이컵, 얼룩 묻은 큐브 |

## 라벨링 규칙

- 불량품에는 `defective` 박스 **하나만** 그립니다. 같은 물체에 정상 클래스 박스를 겹쳐 그리지 않습니다.
- 박스는 물체에 꼭 맞게 그리고, 사진에 보이는 물체는 빠짐없이 모두 표시합니다.
- 물체가 없는 사진은 `e` 키로 배경으로 표시합니다. 오탐을 줄이는 데 도움이 됩니다.

## 촬영 요령

- 실제로 검출할 환경과 같은 카메라, 조명, 배경에서 찍습니다.
- 각도, 위치, 거리를 바꿔 가며 찍고, 한 장에 여러 물체가 함께 나오는 사진도 포함합니다.
- 불량 유형 4가지를 모두 고르게 찍습니다. 얼룩이나 찌그러짐은 카메라에 보이도록 찍어야 학습됩니다.
- 비슷한 사진을 연달아 찍기보다 물체를 옮겨 가며 찍습니다.

## 절차

```
cd ~/git/USB-Camera/custom

# 1. 사진 수집 (카메라 창: SPACE 저장, q 종료 / 터미널: Enter 저장, q 후 Enter 종료)
python3 collect.py

# 2. 라벨링 (사용법은 실행 시 터미널에 표시)
python3 label.py

# 3. 라벨 검사, 클래스별 개수 확인, 학습 목록 생성
python3 prepare.py
```

SSH로 접속했다면 1, 2번 명령 앞에 `DISPLAY=:1`을 붙입니다. 창은 Jetson 모니터에 뜹니다.
다른 카메라 앱이 실행 중이면 `collect.py`가 카메라를 열지 못합니다.

## 학습

사전 학습 가중치를 받은 뒤, 이 폴더에서 Darknet을 실행합니다.

```
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
~/git/darknet/darknet detector train obj.data yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map
```

결과 가중치는 `backup/`에 저장됩니다. 경로가 모두 이 폴더 기준이므로, 폴더째 복사해 GPU가 더 좋은 PC에서 학습해도 됩니다.
다른 PC에서는 그 PC에서 빌드한 Darknet을 사용합니다.
