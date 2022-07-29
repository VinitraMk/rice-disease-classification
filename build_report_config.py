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
                    'details_file': fn,
                    'image_file': visfn
                }
                all_files.append(obj)
    return all_files

if not(os.getenv('ROOT_DIR')):
    os.environ['ROOT_DIR'] = os.getcwd()

all_files = get_files_list()
model_config = dict()
with open('./ergconfig.json') as fp:
    model_config = json.load(fp)
    model_config['files_list'] = all_files

with open('./ergconfig.json', 'w') as fp:
    json.dump(model_config, fp)

images_dir = f'{reports_dir}/images'
if (not(os.path.exists(images_dir))):
    os.mkdir(images_dir)

#cmdserver = 'npx makereport'
#subprocess.call(cmdserver, shell=True)
#cmdserver = f'python -m http.server 5000 -d {reports_dir}'
#subprocess.call(cmdserver)