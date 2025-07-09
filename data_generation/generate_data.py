import configparser
import os
import shutil
import uuid
import json
import tqdm

from simulator.random_state_generator import RandomStateGenerator
from simulator.descent import calculate_firing_solution

CONFIG_NAME = '../generation_params.conf'

CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_NAME)

def check_empty_or_clear_prompt(path):

    if not os.path.exists(path):
        os.makedirs(path)
        return


    if not os.path.isdir(path):
        raise ValueError("Specfied path is an existing file, not a directory. I don't know what to do with it!")
    
    if os.listdir(path):

        print(f'Directory specified: {path} currently has files in it. Should I clear it before proceeding?')

        yn = ''

        while yn == '' or yn.lower() not in ('y', 'n'):
            yn = input('y/n : ')

        if yn == 'y':
            shutil.rmtree(path)
            os.makedirs(path)

if __name__ == '__main__':
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the configuration file
    config.read(CONFIG_PATH)

    # Access sections and keys
    gun = config['Gun']['use_gun']

    if gun == 'All':
        gun = None

    amount = int(config['Params']['amount'])

    output_path = config['Params']['output_path']

    distance_deviation_scalar = float(config['Params']['distance_deviation_scalar'])
    velocity_std_dev = float(config['Params']['velocity_std_dev'])
    altitude_std_dev = float(config['Params']['altitude_std_dev'])
    minimum_distance = float(config['Params']['minimum_distance'])

    learning_rate = float(config['Descent']['learning_rate'])
    tolerance = float(config['Descent']['tolerance'])
    max_iterations = float(config['Descent']['max_iterations'])

    check_empty_or_clear_prompt(output_path)

    # Start the generation. 

    generated = 0

    state_generator = RandomStateGenerator(
        distance_deviation_scalar = distance_deviation_scalar, 
        velocity_std_dev = velocity_std_dev, 
        altitude_std_dev = altitude_std_dev,
        minimum_distance = minimum_distance
    )

    progress_bar = tqdm.tqdm(total = amount)

    while generated < amount:

        test_case = state_generator.generate(gun)

        try:
            solution = calculate_firing_solution(test_case)

            if solution is not None:

                az, el, time = solution

                generated += 1
                progress_bar.update(1)

                test_case['solution_az']   = az
                test_case['solution_el']   = el
                test_case['solution_time'] = time

                filename = str(uuid.uuid4()) + '.json'

                file_path = os.path.join(output_path, filename)

                with open(file_path, 'w') as file:
                    json.dump(test_case, file)

        except (OverflowError, ValueError):
            pass

