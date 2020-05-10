import json 
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import os

name_model = 'dla_34'
threshold = 0.3

count = 0
name_folder = 'json_'+ name_model
video_folder = 'anno_videos_' + name_model
ls_dir = os.listdir(name_folder)

os.system('mkdir {}'.format(video_folder))

# lsd = *.json
for lsd in ls_dir:
	# convert old format to new format
	# {'image_name':[list of bbox]}
    dic_result_demo = {}
    print('open {}/{}.json'.format(name_folder, lsd))
    with open('{}/{}'.format(name_folder, lsd),'r') as f:
        file_content = json.load(f)
        for ob in file_content:
            
            image_name = ob['image_name']
            if image_name not in dic_result_demo:
                dic_result_demo[image_name] = []
                # count+=1
            dic_result_demo[image_name].append({
                "category_id": ob["category_id"], 
                "bbox": ob["bbox"], 
                "score": ob["score"]
            })

    # use new format to draw
    os.system('mkdir {}/{}'.format(video_folder, lsd.split('.')[0]))
    for imgtmp, lsbbox in dic_result_demo.items():
        img= 'extracted_frames/{}/{}'.format(lsd.split('.')[0], imgtmp.split('/')[-1])
        print('+  drawing {}'.format(img))
        source_img = Image.open(img).convert("RGB")
        draw = ImageDraw.Draw(source_img)
        for ob in lsbbox:
            if ob['score'] >= threshold:
                bbxs = ob['bbox']
                draw.rectangle(((bbxs[0], bbxs[1]), (bbxs[0]+bbxs[2], bbxs[1]+bbxs[3])),outline='red', width=5)
        source_img.save('{}/{}/{}'.format(video_folder,lsd.split('.')[0],img.split('/')[-1]))
        print('saved {}/{}/{}'.format(video_folder,lsd.split('.')[0],img.split('/')[-1]))
        
# print(count)
# dic_result_demo
