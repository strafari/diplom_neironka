# Горячие клавиши в окне:
#      •  c – переключить камеру (по кругу)
#      •  r – нарисовать ОДИН новый ROI (старые стираются)
#      •  a – добавить ещё один ROI
#      •  d – удалить ROI, под которым сейчас центр окна
#      •  s – сохранить текущие ROI в seats.json
#      •  Esc / q – выйти


from __future__ import annotations
import requests
import os
import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon
import json, pathlib, itertools        # для seats.json
from shapely.geometry import Point     # проверка курсор внутри ROI
import time, requests

BASE = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
# BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")  # URL FastAPI
#SEAT_ID     = int(os.getenv("SEAT_ID", ))                      # id места в БД
BACKEND_URL = f"{BASE}/device"
DEVICE_KEY  = os.getenv("DEVICE_KEY", "ojyntHWGrul_idmZAJWpG8osDdL56QgVpZ6IcuxgwwY=")                      
SEND_EVERY  = 1 

ENABLE_GUI = os.getenv("ENABLE_GUI", "0") == "1"

def fetch_seat_ids() -> list[int]:
    url = f"{BASE}/seats"                 # фактически /device/seats
    headers = {"X-Device-Key": DEVICE_KEY}
    while True:
        try:
            print(f"[DEBUG] GET {url} headers={headers}")
            r = requests.get(url, headers=headers, timeout=2)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                return [item["seat_id"] for item in data]
            return data
        except requests.exceptions.RequestException as e:
            print(f"[WARN] fetch_seat_ids failed ({e}), retry in 2s…")
            time.sleep(2)

def safe_imshow(winname: str, img):
    if ENABLE_GUI:
        cv2.imshow(winname, img)

def safe_wait_key(delay: int = 1) -> int:
    return cv2.waitKey(delay) if ENABLE_GUI else -1

# НАСТРОЙКИ 
MODEL_PATH = "yolov8n-pose.pt"
CONF_TH    = 0.30    
KPT_TH     = 0.15     

# доли кадра для roi (xmin, ymin, xmax, ymax)
DEFAULT_ROI_RATIO = (0.25, 0.20, 0.75, 0.80)

AREA_MIN_R, AREA_MAX_R = 0.02, 1.0   # доля площади бокса от кадра
AR_MIN, AR_MAX         = 0.3, 3.0    # соотношение сторон бокса

# пороги пересечения
ROI_BOX_MIN   = 0.30   # ≥30 % бокса внутри ROI
ROI_COVER_MIN = 0.30   # ≥30 % ROI покрыто боксом

DEBUG_RAW    = True
DEBUG_REASON = True
PRINT_EVERY  = 30
MODEL_PATH = "yolov8n-pose.pt"
CONF_TH    = 0.30    
KPT_TH     = 0.15     

# доли кадра для центрального roi (xmin, ymin, xmax, ymax)
DEFAULT_ROI_RATIO = (0.25, 0.20, 0.75, 0.80)

AREA_MIN_R, AREA_MAX_R = 0.02, 1.0   # доля площади бокса от кадра
AR_MIN, AR_MAX         = 0.3, 3.0    # соотношение сторон бокса

DEBUG_RAW    = True
DEBUG_REASON = True
PRINT_EVERY  = 30

#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ 
def list_cameras(max_tested: int = 10) -> list[int]:
    # возвращает индексы устройств, которые удаётся открыть.
    idx = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            idx.append(i)
            cap.release()
    return idx

def switch_camera(cap: cv2.VideoCapture, new_src):
    # переключение камер
    if cap is not None and cap.isOpened():
        cap.release()
    return open_source(new_src)

#  рисует roi
def ask_roi(frame):
    x, y, w, h = map(int, cv2.selectROI("Укажите ROI и Enter", frame, False, False))
    cv2.destroyWindow("Укажите ROI и Enter")
    h_img, w_img = frame.shape[:2]
    return (x / w_img, y / h_img, (x + w) / w_img, (y + h) / h_img)

# нормализация
def ratio2px(shape, ratio):
    h, w = shape[:2]
    x1 = int(ratio[0] * w); y1 = int(ratio[1] * h)
    x2 = int(ratio[2] * w); y2 = int(ratio[3] * h)
    return x1, y1, x2, y2

# сохраняшка
CONFIG = pathlib.Path("seats.json")
_id_gen = itertools.count(1)

def load_saved_rois():
    if CONFIG.exists():
        data = json.loads(CONFIG.read_text(encoding="utf-8"))["seats"]
        last = max((s["seat_id"] for s in data), default=0)
        _ = list(itertools.islice(_id_gen, last))   # сдвигаем счётчик
        return data
    return []

def save_rois(rois):
    CONFIG.write_text(json.dumps({"seats": rois}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"[INFO] ROI сохранены в {CONFIG.resolve()}")


# открывает камеру
def open_source(source: str | int = 0) -> cv2.VideoCapture:
    if isinstance(source, str) and not source.isdigit():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {source}")
        print(f"✓ Открыт источник: {source}")
        return cap

    base_idx = int(source)
    backends = [cv2.CAP_DSHOW] if os.name == "nt" else [0]
    for backend in backends:
        for idx in range(base_idx, base_idx + 5):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"✓ Камера открыта (index={idx}, backend={backend})")
                return cap
    raise RuntimeError("Не удалось открыть камеру")

# автоматически создаёт первый roi
def build_roi_poly(frame_shape: tuple[int, int], roi_ratio: tuple[float, float, float, float]) -> Polygon:
    h, w = frame_shape[:2]
    x1 = int(roi_ratio[0] * w)
    y1 = int(roi_ratio[1] * h)
    x2 = int(roi_ratio[2] * w)
    y2 = int(roi_ratio[3] * h)
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

# возвращает true если находит хотя бы 2 точки скелета (глаза, нос, плечи)
def is_human_kpts(kconf: torch.Tensor) -> bool:
    facial_idx     = [0, 1, 2]
    shoulder_idx   = [5, 6]
    facial_ok   = (kconf[facial_idx]   > KPT_TH).sum() >= 2
    shoulders_ok = (kconf[shoulder_idx] > KPT_TH).sum() == 2
    return facial_ok and shoulders_ok

# отправка статуса места на бэк
def send_status_to_backend(seat_id: int, occupied: bool):
    url = f"{BACKEND_URL}/seats/{seat_id}/status"
    headers = {"X-Device-Key": DEVICE_KEY} if DEVICE_KEY else {}
    payload = {"seat_status": 1 if occupied else 0}
    requests.put(url, json=payload, headers=headers, timeout=2)



# ── сеть: отправить 0 / 1 на сервер ────────────────────
# def send_status_to_backend(seat_id: int, status: int) -> None:
    
    ##PUT /device/seats/{SEAT_ID}/status  {"seat_status": status}

    # url = f"{BACKEND_URL}/device/seats/{seat_id}/status"
    # headers = {"X-Device-Key": DEVICE_KEY} if DEVICE_KEY else {}
    # try:
    #     r = requests.put(url, json={"seat_status": status},
    #                      headers=headers, timeout=2)
    #     r.raise_for_status()
    #     print(f"✓ sent seat {seat_id} → {status}")
    # except Exception as e:
    #     # не прерываем детекцию, если сеть упала
    #     print(f"✗ send error: {e}")

# ОСНОВНАЯ ФУНКЦИЯ 

def main():
    ap = argparse.ArgumentParser("Seat monitor – central ROI human detector")
    ap.add_argument("--source", default=0, help="Camera index, video file or RTSP url (default: 0)")
    ap.add_argument("--roi_ratio", default="0.25,0.20,0.75,0.80", help="ROI as xmin,ymin,xmax,ymax in 0‑1 (default central box)")
    args = ap.parse_args()
    
    
    mouse_pos = {"x": None, "y": None}
    window_name = "Seat monitor - center ROI"
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
           mouse_pos["x"], mouse_pos["y"] = x, y 
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    
    
    cam_list = list_cameras()                   # [0, 1, 2, …]
    if not cam_list:
        raise RuntimeError("Камеры не найдены")

    cur_cam_idx = 0                            # позиция в списке cam_list
    start_src   = int(args.source) if str(args.source).isdigit() else cam_list[0]
    if start_src in cam_list:
        cur_cam_idx = cam_list.index(start_src)
    else:
        cam_list.insert(0, start_src)          

    cap = open_source(cam_list[cur_cam_idx])
    print(f"[INFO] Стартуем с камеры {cam_list[cur_cam_idx]}")
    
    
    seat_ids = fetch_seat_ids()
    raw_saved = load_saved_rois()

    
    if not raw_saved and args.roi_ratio != "ask":
        raw_saved = [{
            "roi": tuple(map(float, args.roi_ratio.split(",")))
        }]


    # если json пуст, то рисует один roi по умолчанию
    saved_rois: list[dict] = []
    for idx, entry in enumerate(raw_saved):
        if idx < len(seat_ids):
            saved_rois.append({
                "seat_id": seat_ids[idx],
                "roi":      entry["roi"]
            })
        else:
            print(f"[WARN] нет DB-seat_id для ROI #{idx}, пропускаем")

    rois: list[dict] = []     
    
    
    def rebuild_rois(shape):
        rois.clear()
        for seat in saved_rois:
            sid = seat["seat_id"] # берет какой то seat id для привязки к блоку
            if sid not in seat_ids:
                print(f"[WARN] пропускаем ROI для несуществующего seat_id={sid}")
                continue
            poly = build_roi_poly(shape, tuple(seat["roi"]))
            rois.append({
                "seat_id": sid,
                "ratio":   tuple(seat["roi"]),
                "poly":    poly,
                "area":    poly.area,
                "prev":    None
            })

    # считывает первый кадр
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр.")
    rebuild_rois(frame.shape)

    torch.set_num_threads(1)
    model = YOLO(MODEL_PATH)
    
    
    # координаты для блоков 
    roi_ratio = tuple(map(float, args.roi_ratio.split(",")))  
    if len(roi_ratio) != 4 or not all(0.0 <= v <= 1.0 for v in roi_ratio):
        raise ValueError("roi_ratio должно быть 4 числа 0‑1, разделённых запятой")

    #torch.set_num_threads(1)
    #model = YOLO(MODEL_PATH)

    ##cap = open_source(args.source)

    # формирует автоматический roi для первого кадра
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Не удалось прочитать первый кадр.")
    
    
    h0, w0 = frame.shape[:2]
    mouse_pos["x"], mouse_pos["y"] = w0 // 2, h0 // 2
    
    
    roi_poly = build_roi_poly(frame.shape, roi_ratio)
    roi_area = roi_poly.area

    frame_id = 0
    
    
    prev_status = None          

    # ЦИКЛ ОБРАБОТКИ 
    while ok:
        frame_id += 1
        h_img, w_img = frame.shape[:2]
        img_area = w_img * h_img

        # yolo ‑инференс
        res = model.predict(
            frame,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            half=torch.cuda.is_available(),
            imgsz=640,
            conf=CONF_TH,
            iou=0.45,
            verbose=False,
        )[0]

        valid_boxes, reasons = [], []

        # фильтр для нахождения именно человека
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            aspect = w / max(h, 1)

            area_r  = (w * h) / img_area
            kconf   = res.keypoints.conf[i]
            kpt_ok  = is_human_kpts(kconf)

            box_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            in_roi = False
            for r in rois:
                inter = r["poly"].intersection(box_poly).area
                overlap_to_box = inter / max(box_poly.area, 1)
                overlap_to_roi = inter / max(r["area"],      1)
                if (overlap_to_box >= ROI_BOX_MIN) or (overlap_to_roi >= ROI_COVER_MIN):
                    in_roi = True
                    break

            fail = []
            if not (AR_MIN < aspect < AR_MAX):
                fail.append("AR")
            if not (AREA_MIN_R < area_r < AREA_MAX_R):
                fail.append("AREA")
            if not kpt_ok:
                fail.append("KPTS")
            if not in_roi:
                fail.append("ROI")

            if fail:
                reasons.append(",".join(fail))
            else:
                valid_boxes.append((x1, y1, x2, y2))

        
        # seat_status = 1 if len(valid_boxes) else 0
        
        # if seat_status != prev_status or frame_id % SEND_EVERY == 0:
        #     send_status_to_backend(r["seat_id"], seat_status)
        #     prev_status = seat_status

        for r in rois:
     # подсчет людей в блоке
            count_in_roi = 0
            for (bx1, by1, bx2, by2) in valid_boxes:
                box_poly = Polygon([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)])
                inter = r["poly"].intersection(box_poly).area
                if (inter / max(box_poly.area, 1) >= ROI_BOX_MIN
                    or inter / max(r["area"], 1) >= ROI_COVER_MIN):
                    count_in_roi += 1

            status = 1 if count_in_roi > 0 else 0
            # посылает в бд если что то поменялось
            if r["prev"] is None or status != r["prev"] or frame_id % SEND_EVERY == 0:
                send_status_to_backend(r["seat_id"], status)
                r["prev"] = status

        # дебаг
        if DEBUG_REASON and frame_id % PRINT_EVERY == 0:
            print(f"[{frame_id}] Отброшено:", reasons)

        # рисовашки
        annotated = frame.copy()
        for (bx1, by1, bx2, by2) in valid_boxes:
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

        # для каждого блока считает людей 
        for r in rois:
            x1, y1, x2, y2 = ratio2px(frame.shape, r["ratio"])
            # подсчёт людей в 1 блоке
            count_in_roi = 0
            for (bx1, by1, bx2, by2) in valid_boxes:
                from shapely.geometry import Polygon as _Polygon
                box_poly = _Polygon([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)])
                inter = r["poly"].intersection(box_poly).area
                if inter / max(box_poly.area, 1) >= ROI_BOX_MIN or inter / max(r["area"], 1) >= ROI_COVER_MIN:
                    count_in_roi += 1
            # меняет цвет рамки на зеленый, если нашел человека
            color = (0, 255, 0) if count_in_roi > 0 else (255, 255, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2),color, 2)
    
            #if count_in_roi > 0:
            #    cv2.putText(annotated, f"People in ROI: {count_in_roi}", (x1 + 4, y1 + 18),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
       # cv2.putText(annotated, f"People in ROI: {len(valid_boxes)}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                               # "c- переключить камеру\n",
                                #"r- новый блок\n")
        cv2.imshow(window_name, annotated)

        if DEBUG_RAW:
            dbg = frame.copy()
            for b in res.boxes.xyxy.cpu().numpy():
                cv2.rectangle(dbg, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 1)
            cv2.imshow("DEBUG raw boxes", dbg)
        
        def ru(ch: str) -> int:
            return ch.encode("cp1251")[0]

        # следующий кадр / выход
        key = cv2.waitKey(1) & 0xFF

        # выход
        if key in (27, ord('q')):
            break

        #  переключить камеру
        elif key == ord('c') or key == ru('с'):
            cur_cam_idx = (cur_cam_idx + 1) % len(cam_list)
            cap = switch_camera(cap, cam_list[cur_cam_idx])
            print(f"[INFO] Переключился на камеру {cam_list[cur_cam_idx]}")


        #  задать ОДИН ROI заново мышью
        elif key == ord('r') or key == ru('к'):
            used  = {s["seat_id"] for s in saved_rois}
            avail = [sid for sid in seat_ids if sid not in used]
            if not avail:
                print("[WARN] нет свободных мест для нового ROI")
            else:
                saved_rois[:] = [{
                    "seat_id": avail[0],
                    "roi":      ask_roi(frame)
                }]
                rebuild_rois(frame.shape)
                print("[INFO] ROI переопределён")

        # добавить ещё один roi
        elif key == ord('a') or key == ru('ф'):
            used  = {s["seat_id"] for s in saved_rois}
            avail = [sid for sid in seat_ids if sid not in used]
            if not avail:
                print("[WARN] нет свободных мест для нового ROI")
            else:
                saved_rois.append({
                    "seat_id": avail[0],
                    "roi":      ask_roi(frame)
                })
                rebuild_rois(frame.shape)
                print("[INFO] ROI добавлен")

        # удаляет roi наведеленим курсока
        
        elif key == ord('d') or key == ru('в'):
            if mouse_pos["x"] is None or mouse_pos["y"] is None:
                print("[WARN] Наведите мышь на ROI и нажмите 'd'")
            else:
                px, py = int(mouse_pos["x"]), int(mouse_pos["y"])
                hit = next((r for r in rois if r["poly"].covers(Point(px, py))), None)
                if hit:
                    saved_rois[:] = [
                        s for s in saved_rois
                        if s["seat_id"] != hit["seat_id"]
                    ]
                    rebuild_rois(frame.shape)
                    print(f"[INFO] ROI seat_id={hit['seat_id']} удалён")
                else:
                    print(f"[WARN] Ни один ROI не содержит точку ({px},{py})")
        # elif key == ord('d') or key == ru('в'):
        #     # Если курсор ещё не перемещался — предупредим пользователя
        #     if mouse_pos["x"] is None or mouse_pos["y"] is None:
        #         print("[WARN] Переместите курсор в окно и наведите на ROI, затем нажмите 'd'")
        #         continue
        #     px, py = int(mouse_pos["x"]), int(mouse_pos["y"])
        #     # covers() учитывает и границу полигона
        #     hit = next((r for r in rois if r["poly"].covers(Point(px, py))), None)
        #     if hit:
        #         saved_rois[:] = [s for s in saved_rois if s["seat_id"] != hit["seat_id"]]
        #         rebuild_rois(frame.shape)
        #         print(f"[INFO] ROI seat_id={hit['seat_id']} удалён")
        #     else:
        #         print(f"[WARN] Ни один ROI не содержит точку ({px}, {py})")

        # сохраняте блоки 
        elif key == ord('s') or key == ru('ы'):
            save_rois(saved_rois)
            print("[INFO] ROI сохранены в seats.json")

        ok, frame = cap.read()
        
        
           
    # отправка изменений каждый кадр
        if frame_id == 1:
            prev_status = None        

 
            
        


    # закывает все окна
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
