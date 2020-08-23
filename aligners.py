import os

from aeneas.executetask import ExecuteTask
from aeneas.task import Task

from utils import read_csv


def aeneas(audio_path, transcription_words, start_index, end_index):
    """[summary]

    Args:
        audio_path ([type]): [description]
        transcription_words ([type]): [description]
        start_index ([type]): [description]
        end_index ([type]): [description]

    Returns:
        [type]: [description]
    """
    root_directory = os.path.abspath(os.path.dirname(__file__))
    transcription_txt_path = "./datasets/transcription.txt"
    aligning_result_path = "./datasets/aligning_result.csv"

    # Output the transcription words into a txt file.
    with open(transcription_txt_path, 'w') as txt:
        for word in transcription_words:
            txt.write(word)
            txt.write('\n')
    
    # Execute the aligning task and get the result in a csv file.
    task = Task(config_string="task_language=eng|is_text_type=plain|os_task_file_format=csv")
    task.audio_file_path_absolute = audio_path
    task.text_file_path_absolute = os.path.join(root_directory, transcription_txt_path)
    task.sync_map_file_path_absolute = os.path.join(root_directory, aligning_result_path)
    ExecuteTask(task).execute()
    task.output_sync_map_file()

    # Calculate the start and end sample index.
    fragments = read_csv(path=aligning_result_path)
    start_index = int(16000 * float(fragments[start_index][1]))
    end_index = int(16000 * float(fragments[end_index][2])) + 1

    os.remove(transcription_txt_path)
    os.remove(aligning_result_path)

    return start_index, end_index
