import os
import atexit
import tempfile
import shutil
import socket
from random import random
from time import sleep, time
import argparse
import subprocess
from datetime import datetime
import math
import yaml

import numpy as np
import pandas as pd

from celery import Celery, states
import drmaa

from vivarium.framework.util import collapse_nested_dict, expand_branch_templates

import logging
_log = logging.getLogger(__name__)

from celery import signals

@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
        pass

def uge_specification(peak_memory, job_name='ceam'):
    #Are we on dev or prod?
    result = str(subprocess.check_output(['qconf', '-ss']))
    if 'cluster-dev.ihme.washington.edu' in result:
        project = None
    elif 'cluster-prod.ihme.washington.edu' in result:
        project = 'proj_cost_effect'
    else:
        raise Exception('This script must be run on the IHME cluster')

    preamble = '-w n -q all.q -l m_mem_free={}G -N {}'.format(peak_memory, job_name)

    # Calculate slot count based on expected peak memory usage and 2g per slot
    num_slots = int(math.ceil(peak_memory/2.5))
    preamble += ' -pe multi_slot {}'.format(num_slots)

    if project:
        preamble += ' -P {}'.format(project)

    return preamble


def init_job_template(jt, peak_memory, broker_url, config_file, worker_log_directory):
    launcher = tempfile.NamedTemporaryFile(mode='w', dir='.', prefix='celery_worker_launcher_', suffix='.sh', delete=False)
    atexit.register(lambda: os.remove(launcher.name))
    launcher.write('''
    {} -A vivarium.framework.celery_tasks worker --without-gossip -Ofair --without-mingle --concurrency=1 -f {} --config {} -n ${{JOB_ID}}.${{SGE_TASK_ID}}
    '''.format(shutil.which('celery'), os.path.join(worker_log_directory, 'worker-${JOB_ID}.${SGE_TASK_ID}.log'), config_file))
    launcher.close()

    jt.workingDirectory = os.getcwd()
    jt.remoteCommand = shutil.which('sh')
    jt.args = [launcher.name]
    sge_cluster = os.environ['SGE_CLUSTER_NAME']
    jt.jobEnvironment = {
                'LC_ALL': 'en_US.UTF-8',
                'LANG': 'en_US.UTF-8',
                'SGE_CLUSTER_NAME': sge_cluster,
            }
    jt.joinFiles=True
    jt.nativeSpecification = uge_specification(peak_memory)
    jt.outputPath=':/dev/null'
    return jt

def get_random_free_port():
    # NOTE: this implementation is vulnerable to rare race conditions where some other process gets the same
    # port after we free our socket but before we use the port number we got. Should be so rare in practice
    # that it doesn't matter.
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def launch_redis(port):
    redis_process = subprocess.Popen(["redis-server", "--protected-mode", "no", "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(redis_process.kill)
    return redis_process

def launch_celery_flower(port, config):
    args = ['celery', 'flower', '-A', 'vivarium.framework.celery_tasks', '--config={}'.format(config), '--port={}'.format(port)]
    print('celery '+' '.join(args))
    flower_process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(flower_process.kill)
    return flower_process

def start_cluster(num_workers, peak_memory, worker_log_directory):
    hostname = socket.gethostname()
    port = get_random_free_port()
    _log.info('Starting Redis Broker at %s:%s', hostname, port)
    broker_process = launch_redis(port)
    broker_url = 'redis://{}:{}'.format(hostname, port)

    port = get_random_free_port()
    _log.info('Starting Redis Result Backend at %s:%s', hostname, port)
    backend_process = launch_redis(port)
    backend_url = 'redis://{}:{}'.format(hostname, port)


    celery_config = tempfile.NamedTemporaryFile(mode='w', dir='.', prefix='celery_worker_config_', suffix='.py', delete=False)
    atexit.register(lambda: os.remove(celery_config.name))
    config = {
        'broker_url': broker_url,
        'result_backend': backend_url,
        'worker_prefetch_multiplier': 1,
        'worker_concurrency': 1,
        'task_acks_late': True,
        'redis_max_connections': 1,
        'redis_socket_connect_timeout': 10,
        'broker_connection_max_retries': 10,
        'broker_transport_options': {'max_connections': 1},
    }
    celery_config.write('\n'.join(['{} = {}'.format(k,repr(v)) for k,v in config.items()]))
    celery_config.flush()
    celery_config.close()
    config_module = os.path.splitext(os.path.basename(celery_config.name))[0]

    port = get_random_free_port()
    _log.info('Starting Celery Flower Monitoring at %s:%s', hostname, port)
    flower_process = launch_celery_flower(port, config_module)

    s=drmaa.Session()
    s.initialize()
    jt=init_job_template(s.createJobTemplate(), peak_memory, broker_url, config_module, worker_log_directory)
    if num_workers:
        job_ids = s.runBulkJobs(jt, 1, num_workers, 1)
        array_job_id = job_ids[0].split('.')[0]
        atexit.register(lambda: s.control(array_job_id, drmaa.JobControlAction.TERMINATE))

    app = Celery(broker=broker_url, backend=backend_url)
    atexit.register(lambda: app.control.broadcast('shutdown'))
    return app


def load_branch_configurations(path):
    with open(path) as f:
        data = yaml.load(f)

    draw_count = data['draw_count']
    assert draw_count <= 1000, "Cannot use more that 1000 draws from GBD"

    branches = expand_branch_templates(data['branches'])

    return draw_count, branches


def write_results(results, directory):
    results.to_hdf(os.path.join(directory, 'output.hdf'), 'data')


def calculate_keyspace(branches):
    keyspace = {k:{v} for k,v in collapse_nested_dict(branches[0])}

    for branch in branches[1:]:
        branch = dict(collapse_nested_dict(branch))
        if set(branch.keys()) != set(keyspace.keys()):
            raise ValueError("All branches must have the same keys")
        for k,v in branch.items():
            keyspace[k].add(v)
    return keyspace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component_configuration_file', type=str, help='Path to component yaml file')
    parser.add_argument('branch_configuration_file', type=str, nargs='?', default=None, help='Path to branch yaml file')
    parser.add_argument('--num_input_draws', '-d', type=int, default=None)
    parser.add_argument('--num_model_draws', type=int, default=None)
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--num_workers', '-w', type=int, default=None)
    parser.add_argument('--max_retries', type=int, default=1, help='Maximum times to retry a failed job')
    parser.add_argument('--peak_memory', type=float, default=3, help='Maximum vmem that each process can use. This determines slots/worker.')
    parser.add_argument('--result_directory', '-o', type=str, default='/home/j/Project/Cost_Effectiveness/CEAM/Results/{config_name}/{launch_time}/')
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--log', '-l', type=str, default='{results_directory}/master.log')

    args = parser.parse_args()
    launch_time = datetime.now()

    config_name = os.path.basename(args.component_configuration_file.rpartition('.')[0])
    results_directory = args.result_directory.format(config_name=config_name, launch_time=launch_time.strftime("%Y_%m_%d_%H_%M_%S"))
    worker_log_directory = os.path.join(results_directory, 'worker_logs')

    os.makedirs(results_directory, exist_ok=True)
    os.makedirs(worker_log_directory, exist_ok=True)

    master_log_file = args.log.format(results_directory=results_directory)

    log_level = logging.ERROR if args.quiet else logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=master_log_file, level=log_level)
    logging.getLogger().addHandler(logging.StreamHandler())

    if args.num_input_draws is None and args.branch_configuration_file is not None:
        input_draw_count, branches = load_branch_configurations(args.branch_configuration_file)
        keyspace = calculate_keyspace(branches)
    elif args.num_input_draws is not None and args.branch_configuration_file is None:
        input_draw_count = args.num_input_draws
        branches = [None]
        keyspace = {}
    else:
        raise ValueError('Must supply one of branch_configuration_file or --num_draws but not both')

    model_draws = range(args.num_model_draws) if args.num_model_draws is not None else [0]

    if args.component_configuration_file.endswith('.yaml'):
        with open(args.component_configuration_file) as f:
            component_config = yaml.load(f)
    else:
        raise ValueError("Unknown components configuration type: {}".format(args.component_configuration_file))

    run_configuration = component_config['configuration'].get('run_configuration', {})
    run_configuration['results_directory'] = results_directory
    component_config['configuration']['run_configuration'] = run_configuration

    np.random.seed(123456)

    if input_draw_count < 1000:
        input_draws = np.random.choice(range(1000), input_draw_count, replace=False)
    else:
        input_draws = range(1000)


    keyspace['input_draw'] = input_draws
    keyspace['model_draw'] = model_draws

    with open(os.path.join(results_directory, 'keyspace.yaml'), 'w') as f:
        yaml.dump(keyspace, f)

    _log.info('Starting jobs. Results will be written to: {}'.format(results_directory))

    jobs = []
    for branch_config in branches:
        for model_draw_num in model_draws:
            for input_draw_num in input_draws:
                jobs.append((int(input_draw_num), int(model_draw_num), component_config, branch_config, worker_log_directory))
    np.random.shuffle(jobs)

    if args.group:
        i,t = [int(x) for x in args.group.split(':')]
        stride = int(len(jobs)/t)
        jobs = jobs[i*stride:i*stride+stride]

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = len(jobs)

    celery_app = start_cluster(num_workers, args.peak_memory, worker_log_directory)

    futures = {}
    for job in jobs:
        futures[celery_app.send_task('vivarium.framework.celery_tasks.worker', job, acks_late=True)] = job

    results = pd.DataFrame()
    results_dirty = False
    while futures:
        sleep(3)
        celery_stats = celery_app.control.inspect().stats()
        if celery_stats:
            workers = len(celery_stats)
        else:
            workers = 0
        _log.info('Pending: {} (active workers: {})'.format(len(futures), workers))

        new_futures = {}
        for future, args in futures.items():
            if future.state == states.SUCCESS:
                results = results.append(pd.read_json(future.result))
                results_dirty = True
            elif future.state == states.FAILURE:
                exception = future.result
                tb = future.traceback
                _log.info('Failure in job %s\n%s', args, tb)
            else:
                new_futures[future] = args
        futures = new_futures

        if results_dirty:
            write_results(results, results_directory)
            results_dirty = False


if __name__ == '__main__':
    main()
