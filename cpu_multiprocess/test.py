import os.path
import pathlib
import io
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count

"""
Credit to: https://gist.github.com/kaspermunch/64e11cd21e3953295e149e75bee51733
"""


def test_multiprocess(input_dir, save_dir):
    print(f"input_dir: {input_dir}. output_dir: {save_dir}")


def process_batch(data_list, input_dir, output_dir):
    for d in tqdm(data_list):
        save_dir = os.path.join(output_dir, d)
        input_dir = os.path.join(input_dir, d)
        test_multiprocess(frame_dir, save_dir)
        # batch_detect_mouth(frame_dir, save_dir)


if __name__ == "__main__":
    input_dir = "input_dir"
    output_dir = "output_dir"
    
    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])

    dirs = np.array_split(
        [f"test_{i}" for i in range(40)],
        number_of_cores)

    print(number_of_cores)

    args = []
    for i in range(number_of_cores):
        args.append((dirs[i], input_dir, output_dir))

    # multiprocssing pool to distribute tasks to:
    with Pool(number_of_cores) as pool:
        # distribute computations and collect results:
        results = pool.starmap(process_batch, args)

    pool.close()
    pool.join()