import json
import os

class_map={}

def convert_bbox_to_yolo(box2d, img_width, img_height):
    x1, y1, x2, y2 = box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']
    x_center = (x1+x2)/2/img_width
    y_center = (y1+y2)/2/img_height
    width = (x2-x1)/img_width
    height = (y2-y1)/img_height
    return x_center, y_center, width, height

def conver_bdd_to_yolo(json_path, output_dir, img_width, img_height):
    with open(json_path, 'r') as f:
        data=json.load(f)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in data:
        img_name = item['name']
        labels = item.get('labels', [])
        txt_path = os.path.join(output_dir, img_name.split('.')[0]+'.txt')

        with open(txt_path, 'w') as txt_file:
            for label in labels:
                category = label['category']
                if category not in class_map:
                    class_map[category]=len(class_map)
                class_id = class_map[category]
                box2d = label.get('box2d')
                if not box2d:
                    continue

                x_center, y_center, width, height = convert_bbox_to_yolo(box2d, img_width, img_height)
                txt_file.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
    print(f'Convert {json_path} to YOLO format in {output_dir} successfully!')              
    print(f'Class map: {class_map}')

json_path='bdd100k/labels/det_20/det_val.json'
output_dir='bdd100k/labels/val'
img_width, img_height = 1280, 720

conver_bdd_to_yolo(json_path, output_dir, img_width, img_height)