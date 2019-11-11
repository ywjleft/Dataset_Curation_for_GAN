# run this code and visit by browser https://$your-server-name-or-ip$:5001/main
from flask import Flask, request, render_template, jsonify
#import ssl
import time
import argparse
import os
import threading

parser = argparse.ArgumentParser()
parser.add_argument('-experiment_name', default='{}_analyze'.format(int(time.time())))
parser.add_argument('-gpuid', default='0')
parser.add_argument('-enable_simul', type=bool, default=False)
parser.add_argument('-datatype', default='face')

argsk, argsu = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = argsk.gpuid

from config import outputroot
outputpath = os.path.join(outputroot, argsk.experiment_name)
os.makedirs(outputpath, exist_ok=True)

#context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
#context.load_cert_chain('cert.cert', 'key.key')
app = Flask(__name__)
app.secret_key = b'\x05:#\xe9\xb0fe\xe7\x96\x0fi\xeb\x7fF\xc1\xda'

currentuid = '{}'.format(int(time.time()))

if argsk.datatype in ['face', 'bedroom']:
    from curation_system import curation_system
    system = curation_system(len(argsk.gpuid.split(',')), argsk.datatype, currentuid, outputpath, argsk.enable_simul)
elif argsk.datatype in ['wood', 'metal', 'stone']:
    from curation_system_texture import curation_system
    system = curation_system(len(argsk.gpuid.split(',')), argsk.datatype, currentuid, outputpath, argsk.enable_simul)


def background_calculation(userinput):
    global dcache
    system.sendResult(userinput)
    d0 = system.getQuery()['d0']
    dcache = {**{'round': system.round, 'tips': 'Round{}, select examples you prefer.'.format(system.round)}, **d0}


@app.route('/calculate', methods=['POST'])
def calculate():
    userinput = request.get_data(as_text=True)
    if argsk.enable_simul:
        global dcache, thd
        if system.round == 0:
            thd = None
            d0 = system.getQuery()['d0']
            d = {**{'round': system.round, 'tips': 'Round{}, select examples you prefer.'.format(system.round)}, **d0}
            d1 = system.getQuery()['d0']
            dcache = {**{'round': system.round, 'tips': 'Round{}, select examples you prefer.'.format(system.round)}, **d1}
            return jsonify(d)
        else:
            if not thd is None:
                thd.join()
            dcurrent = dcache
            thd = threading.Thread(target=background_calculation, args = (userinput,))
            thd.start()
            return jsonify(dcurrent)

    else:
        if system.round > 0:
            print('{}, {}'.format(len(userinput), userinput))
            system.sendResult(userinput)

        d0 = system.getQuery()['d0']
        d = {**{'round': system.round, 'tips': 'Round{}, select examples you prefer.'.format(system.round)}, **d0}
        return jsonify(d)


@app.route('/main', methods=['GET'])
def main():
    return render_template('index.html')


#app.run(host='0.0.0.0', port='5001', debug=False, ssl_context=context)
app.run(host='0.0.0.0', port='5001', debug=False)