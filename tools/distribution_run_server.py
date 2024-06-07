from flask import Flask, render_template, request, jsonify
import datetime
import time
import threading

app = Flask(__name__)

# 假设正在运行的程序信息存在一个全局变量中
running_programs = []
server_infos = []
left_assign_time = 60

# 输入页面路由，渲染 input.html 模板
@app.route('/')
def input_page():
    return render_template('input.html', running_programs=running_programs, server_infos=server_infos, left_assign_time=left_assign_time)

# 处理用户提交的程序信息并进行调度
@app.route('/submit', methods=['POST'])
def submit_program():
    command = request.form.get('command')

    command = command.split('##')

    gpu = float(request.form.get('gpu'))/10
    mem = float(request.form.get('mem'))

    for _command in command:
        running_programs.append({'command': _command.replace('\n',''), 'position': '尚未分配', 'gpu': gpu, 'mem': mem})
    return jsonify({'message': '程序已分配节点运行'})

# 处理用户提交的程序信息并进行调度
@app.route('/clear', methods=['GET'])
def clear_program():
    running_programs.clear()
    return jsonify({'message': '已经清空已提交程序列表'})

# 处理用户提交的程序信息并进行调度
@app.route('/update_servers', methods=['POST'])
def update_server_info():
    global running_programs
    server_name = request.form.get('name')
    server_gpus = eval(request.form.get('gpu_stat'))
    server_gpus = [float(g) for g in server_gpus]
    server_mem = float(request.form.get('mem_stat',12))
    find = False
    for i in range(len(server_infos)):
        if server_infos[i]['name'] != server_name:
            continue
        server_infos[i]['gpu_stat'] = server_gpus
        server_infos[i]['mem_stat'] = server_mem
        server_infos[i]['time'] = datetime.datetime.now()
        find = True
        break

    if not find:
        i = len(server_infos)
        server_infos.append({'name':server_name, 
                             'gpu_stat': server_gpus,
                             'mem_stat': server_mem,
                             'time':datetime.datetime.now()})

    # print(server_infos)
    cur_need_to_run = []
    _running_programs = []
    for j in range(len(running_programs)):
        name = running_programs[j]['position']
        if name != '尚未分配':
            name,gpu_id = tuple(name.split('##'))
            mem = running_programs[j]['mem']
            gpu = running_programs[j]['gpu']

            if name == server_name:
                cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup {running_programs[j]['command']} &"
                server_infos[i]['mem_stat'] -= mem
                server_infos[i]['gpu_stat'][int(gpu_id)] -= gpu
                cur_need_to_run.append(cmd)
            else:
                _running_programs.append(running_programs[j])
        else:
            _running_programs.append(running_programs[j])
    
    running_programs = _running_programs
    return jsonify({'message':cur_need_to_run})


@app.route('/left_assign_time', methods=['GET'])
def _left_assign_time():
    return jsonify({'message':left_assign_time})

def assign():
    global left_assign_time
    while True:
        for i in range(60):
            left_assign_time -= 1
            time.sleep(1)

        js = list(range(len(server_infos)))
        js.sort(key=lambda x:min(sum(server_infos[x]['gpu_stat']) / 1.5, server_infos[x]['mem_stat']/3), reverse=True)
        
        for _j in range(len(server_infos)):
            j = js[_j]
            for i in range(len(running_programs)):
                if running_programs[i]['position'] != '尚未分配':
                    continue
                ks = list(range(len(server_infos[j]['gpu_stat'])))
                ks.sort(key=lambda x:server_infos[j]['gpu_stat'][x], reverse=True)
                k = ks[0]
                if server_infos[j]['gpu_stat'][k] > running_programs[i]['gpu'] and server_infos[j]['mem_stat'] > running_programs[i]['mem']:
                    server_infos[j]['gpu_stat'][k] -= running_programs[i]['gpu']
                    server_infos[j]['mem_stat'] -= running_programs[i]['mem']
                    running_programs[i]['position'] = server_infos[j]['name'] + '##' + str(k)

        left_assign_time = 60

if __name__ == '__main__':
    thread1 = threading.Thread(target=assign)
    thread1.start()
    app.run(debug=False,host='0.0.0.0', port=5000)
    thread1.join()
