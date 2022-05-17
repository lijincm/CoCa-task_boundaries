import os
def run_exp(buffer):
    print(buffer)
    the_command = 'python3 main.py' \
        + ' --load_best_args'   \
        + ' --buffer_size ' + str(buffer) \
        + ' --model coca'\
        + ' --dataset seq-tinyimg'\
        + ' --csv_log'
    os.system(the_command)
    print('tiny')
for i in range(0,1):
    run_exp(200)
    run_exp(500)
    



