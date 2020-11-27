import argparse

from PIL.ImageEnhance import Color

from yolov3.models import *
from yolov3.utils.bbox_iou import calculate_iou
from yolov3.utils.datasets import *
from yolov3.utils.utils import *
from yolov3.yolo_cfg import yolo_cfg, yolo_names, yolo_last, img_size, yolo_images, yolo_pred_output


def detect():
    imgsz = opt.img_size
    out, source, weights, save_txt = opt.output, opt.source, opt.weights, opt.save_txt

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if not os.path.exists(out):
        os.makedirs(out, exist_ok=True)

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    model.load_state_dict(torch.load(weights, map_location=device)['model'])

    # Eval mode
    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred,
                                   opt.conf_thres,
                                   opt.iou_thres,
                                   multi_label=True,
                                   classes=opt.classes)

        pred = filter_bbox(pred)
        # Process detections
        for i, det in enumerate(pred):
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                file = open(save_path[:save_path.rfind('.')] + '.txt', 'w')
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        file.write(f'{int(cls)},{xmin},{ymin},{xmax},{ymax},{round(conf.item(), 4)},{names[int(cls)]}\n')

                    if save_img:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=None, color=(255,255,0))
                file.close()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


def filter_bbox(pred):
    rst = []
    for i, p in enumerate(pred):
        if p is not None and len(p):
            flag = np.ones(len(p))
            for j in range(len(p)):
                for k in range(j + 1, len(p)):
                    iou = calculate_iou(p[j], p[k])
                    if iou > 0:
                        if p[j][4] < p[k][4]:
                            flag[j] = 0
                            flag[k] = 1
                        else:
                            flag[k] = 0
                            flag[j] = 1
            rst.append(p[flag == 1])
    return rst


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=yolo_cfg, help='*.cfg path')
    parser.add_argument('--names', type=str, default=yolo_names, help='*.names path')
    parser.add_argument('--weights', type=str, default=yolo_last, help='weights path')
    parser.add_argument('--source', type=str, default=yolo_images, help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default=yolo_pred_output, help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=img_size, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--save_txt', default=True, action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', default=['0'], type=int, help='filter by class')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        detect()
