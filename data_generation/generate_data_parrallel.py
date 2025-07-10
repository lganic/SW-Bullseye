import configparser
import os
import shutil
import uuid
import json
import tqdm
import concurrent.futures

from simulator.random_state_generator import RandomStateGenerator
from simulator.descent import calculate_firing_solution

CONFIG_NAME = '../params.conf'
CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_NAME)

def check_empty_or_clear_prompt(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    if not os.path.isdir(path):
        raise ValueError("Specified path is a file, not a directory.")
    if os.listdir(path):
        yn = ''
        print(f'Directory {path} has files. Clear it?')
        while yn.lower() not in ('y', 'n'):
            yn = input('y/n : ')
        if yn == 'y':
            shutil.rmtree(path)
            os.makedirs(path)

# Worker function with retry loop
def generate_valid_case(params):
    import random  # needed if your generator uses it

    (
        distance_deviation_scalar,
        velocity_std_dev,
        altitude_std_dev,
        minimum_distance,
        gun,
        learning_rate,
        tolerance,
        max_iterations
    ) = params

    generator = RandomStateGenerator(
        distance_deviation_scalar=distance_deviation_scalar,
        velocity_std_dev=velocity_std_dev,
        altitude_std_dev=altitude_std_dev,
        minimum_distance=minimum_distance
    )

    while True:
        try:
            test_case = generator.generate(gun)
            solution = calculate_firing_solution(test_case)
            if solution is None:
                continue
            az, el, time = solution
            test_case.update({
                'solution_az': az,
                'solution_el': el,
                'solution_time': time
            })
            return test_case
        except (OverflowError, ValueError):
            continue

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    gun = config['Gun']['use_gun']
    if gun == 'All':
        gun = None

    amount = config.getint('Params', 'amount')
    output_path = config.get('Params', 'output_path')

    distance_deviation_scalar = config.getfloat('Params', 'distance_deviation_scalar')
    velocity_std_dev = config.getfloat('Params', 'velocity_std_dev')
    altitude_std_dev = config.getfloat('Params', 'altitude_std_dev')
    minimum_distance = config.getfloat('Params', 'minimum_distance')

    learning_rate = config.getfloat('Descent', 'learning_rate')
    tolerance = config.getfloat('Descent', 'tolerance')
    max_iterations = config.getfloat('Descent', 'max_iterations')

    check_empty_or_clear_prompt(output_path)

    params = (
        distance_deviation_scalar,
        velocity_std_dev,
        altitude_std_dev,
        minimum_distance,
        gun,
        learning_rate,
        tolerance,
        max_iterations
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        progress = tqdm.tqdm(total=amount)
        futures = [executor.submit(generate_valid_case, params) for _ in range(amount)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            filename = f"{uuid.uuid4()}.json"
            filepath = os.path.join(output_path, filename)
            with open(filepath, 'w') as f:
                json.dump(result, f)
            progress.update(1)
