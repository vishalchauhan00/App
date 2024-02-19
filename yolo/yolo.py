from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import json

def detect_objects(image_path):
    # YOLO model loading code
    model = YOLO('D:\Food Detection\weights_22_classes.pt')

    # Perform object detection and get the result image
    results = model.predict(image_path,imgsz=640)

    result_image_path = f'static/uploads/result_{Path(image_path).name}'

    objects = None

    # Show the results
    for r in results:
        '''
        r.names = {
            0: 'Aloo Gobhi (95 cal per bowl)',
            1: 'Aloo Masala (174 cal per serving)',
            2: 'Bhatura (136 cal per unit)',
            3: 'Bhindi Sabji (73 cal per bowl)',
            4: 'Biryani (187 cal per bowl)',
            5: 'Milk Tea (73 cal per cup)',
            6: 'Chole (173 cal per bowl)',
            7: 'Coconut Chutney (307 cal per bowl)',
            8: 'Dal Fry (127 cal per bowl)',
            9: 'Dosa (147 cal per unit)',
            10: 'Aloo Gravy (112 cal per bowl)',
            11: 'Fish Curry (221 cal per serving)',
            12: 'Ghevar (74 cal per unit)',
            13: 'Dhaniya Chutney (5 cal per spoon)',
            14: 'Gulab Jamun (175 cal per unit)',
            15: 'Idli (73 cal per unit)',
            16: 'Jalebi (66 cal per unit)',
            17: 'Kebab (581 cal per unit)',
            18: 'Kheer (305 cal per bowl)',
            19: 'Shahi Kulfi (84 cal per scoop)',
            20: 'Lassi (202 cal per glass)',
            21: 'Mutton Curry (298 cal per serving)',
            22: 'Onion Pakora (135 cal per 50g)',
            23: 'Palak Paneer (177 cal per bowl)',
            24: 'Poha (137 cal per bowl)',
            25: 'Rajma Curry (207 cal per serving)',
            26: 'Ras Malai (331 cal per unit)',
            27: 'Samosa (276 cal per unit)',
            28: 'Shahi Paneer (225 cal per bowl)',
            29: 'Rice (120 cal per bowl)'
        }
        '''
        r.names = {
            0: 'Indian Bread (85-448 cal per unit)',
            1: 'Rasgulla (106 cal per unit)',
            2: 'Biryani (187 cal per bowl)',
            3: 'Uttapam (227 cal per unit)',
            4: 'Paneer Dish (225 cal per bowl)',
            5: 'Poha (137 cal per bowl)',
            6: 'Khichdi (125 cal per bowl)',
            7: 'Egg Omelette (127 cal per egg)',
            8: 'Rice (120 cal per bowl)',
            9: 'Dal Makhani (278 cal per serving)',
            10: 'Rajma Tadka (121 cal per bowl)',
            11: 'Puri (134 cal per unit)',
            12: 'Chole (173 cal per bowl)',
            13: 'Dal Fry (127 cal per bowl)',
            14: 'Sambar (114 cal per bowl)',
            15: 'Papad (29 cal per unit)',
            16: 'Gulab Jamun (175 cal per unit)',
            17: 'Idli (73 cal per unit)',
            18: 'Vada (155 cal per unit)',
            19: 'Dosa (147 cal per unit)',
            20: 'Fries (365 cal per serving)',
            21: 'Burger (391 cal per unit)'
        }
        j = r.tojson()
        j_dict = json.loads(j)
        obj_dict = dict()
        for i in j_dict:
            if i['name'] not in obj_dict.keys():
                obj_dict[i['name']] = 1
            else:
                obj_dict[i['name']] += 1
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.save(result_image_path)  # save image

    # Return the result image path
    return result_image_path, obj_dict
