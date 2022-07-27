from bs4 import BeautifulSoup
from helper.utils import get_config
import os
import re
import json
import subprocess

reports_dir = './reports'
report_template = './reports/dependencies/standard-report.html'
out_path = './reports/index.html'

def get_files_list():
    config = get_config()
    exp_logs = os.listdir(config["experimental_output_path"])
    vis_files = os.listdir(config["visualizations_path"])
    all_files = []
    for fn in exp_logs:
        s = re.findall('\d', fn)
        visfn = ''
        if len(s) != 0:
            datestr = f'{"".join(s[:8])}-{"".join(s[8:])}'
            _visfn = [x for x in vis_files if datestr in x]
            if len(_visfn) != 0:
                visfn = _visfn[0]
                obj = {
                    'experimental_log_path': f"{config['experimental_output_path']}/{fn}",
                    'visualizations_path': f"visualizations/{visfn}"
                }
                all_files.append(obj)
    return all_files

with open(report_template, 'a+') as fp:
    fp.seek(0)
    doc = fp.read()
    soup = BeautifulSoup(doc, 'html.parser')
    root = soup.find(id="root")
    grid = soup.new_tag('div', attrs={'class': "eog-grid"})
    all_files = get_files_list()
    for fn in all_files:
        img_path = fn['visualizations_path']
        with open(fn['experimental_log_path'], 'r') as fpo:
            exp_attrs = json.load(fpo)
        acc = exp_attrs['average_accuracy'] if 'average_accuracy' in exp_attrs else exp_attrs['final_accuracy']
        loss = exp_attrs['average_loss']
        model_params = json.loads(exp_attrs['model_params'])
        batch_size = model_params['batch_size']
        learning_rate = model_params['lr']
        div = soup.new_tag('div' , attrs = { 'class': 'eog-card-img' })
        img_tag = BeautifulSoup(f'<img src="{img_path}" class="eog-card-img__img"/>', 'html.parser')
        caption_tag1 = BeautifulSoup(f'<label class="eog-card-img__caption">Final accuracy: {acc}</label>', 'html.parser')
        caption_tag2 = BeautifulSoup(f'<label class="eog-card-img__caption">Average Loss: {loss}</label>', 'html.parser')
        caption_tag3 = BeautifulSoup(f'<label class="eog-card-img__caption">Learning rate: {learning_rate}</label>', 'html.parser')
        caption_tag4 = BeautifulSoup(f'<label class="eog-card-img__caption">Batch size: {batch_size}</label>', 'html.parser')
        div.append(img_tag)
        div.append(caption_tag1)
        div.append(caption_tag2)
        div.append(caption_tag3)
        div.append(caption_tag4)
        grid.append(div)
    root.append(grid)
with open(out_path, 'w+') as fp:
    fp.write(str(soup))

cmd = f'python -m http.server 5000 -d {reports_dir}'
subprocess.call(cmd)