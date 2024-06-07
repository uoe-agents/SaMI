
from bs4 import BeautifulSoup


import base64
import io
import os
from PIL import Image
	
def base64_to_image(image_base, image_path):
    '''
    将base64编码转换为图片
    :param image_base: base64编码
    :param image_path: 图片路径
    :return:
    '''
    try:
        bytes = image_base.split(',')[-1]
        bytes = base64.b64decode(bytes)  # 将base64解码为bytes
        image = io.BytesIO(bytes)
        image = Image.open(image)
        image.save(image_path)
    except:
        print(image_path, ':error')

def extract_image(html_file):
    output_dir = html_file.replace('.html','')
    
    with open(html_file, 'r', encoding='utf-8') as f:
        data = f.read()
        soup = BeautifulSoup(data,'html.parser')
        table = soup.findAll('table')[0]
        data = {}
        if not os.path.exists('image'):
            os.mkdir('image')

        for i, child in enumerate(table.children):
            if child=='\n':
                continue
            if i == 1:
                continue
            
            child = list(child.children)
            child = list(filter(lambda x:x!='\n',child))
            env = f'image_{output_dir}/' + child[0].get_text().strip()
            method = child[1].get_text().strip().split(';')[0]
            image = list(child[-7].children)[1].attrs['src']
            if not os.path.exists(f'image_{output_dir}'):
                os.mkdir(f'image_{output_dir}')
            if not os.path.exists(env):
                os.mkdir(env)
            base64_to_image(image, os.path.join(env, method + '-tsne.jpg'))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--html_file', default='output.html')
args = parser.parse_args()

ans = extract_image(args.html_file)


