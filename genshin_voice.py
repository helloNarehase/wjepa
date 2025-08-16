import os
import io
import numpy as np
import soundfile as sf
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

# Load the dataset in streaming mode to avoid downloading everything at once
# 모든 것을 한 번에 다운로드하지 않고 데이터를 스트리밍하기 위함입니다.
dataset = load_dataset('simon3000/genshin-voice', split='train', streaming=True)

# Filter the dataset for Korean voices of Ganyu with a non-empty transcription
# 한국어 Ganyu 음성 중 스크립션이 있는 데이터를 필터링합니다.
korean_ganyu = dataset.filter(
    lambda voice: voice['language'] == 'Korean' and voice['speaker'] == 'Ganyu' and voice['transcription'] != ''
)

# Create a folder to store the output files
# 출력 파일을 저장할 폴더를 생성합니다.
ganyu_folder = 'ganyu'
os.makedirs(ganyu_folder, exist_ok=True)

def process_voice(voice, i):
    """
    Processes a single voice clip and saves the audio and transcription.
    단일 음성 클립을 처리하고 오디오 및 스크립션을 저장합니다.
    """
    try:
        # Check if the 'audio' dictionary has 'bytes' key
        # 'audio' 딕셔너리에 'bytes' 키가 있는지 확인합니다.
        if 'bytes' in voice['audio']:
            audio_bytes = voice['audio']['bytes']

            # soundfile.read() 함수는 오디오 데이터와 함께 샘플링 레이트도 반환합니다.
            # The soundfile.read() function returns both the audio data and the sampling rate.
            audio_data, sampling_rate = sf.read(io.BytesIO(audio_bytes))

            # Define the paths for the new audio and transcription files
            # 새 오디오 및 스크립션 파일의 경로를 정의합니다.
            audio_path = os.path.join(ganyu_folder, f'{i}_audio.wav')
            transcription_path = os.path.join(ganyu_folder, f'{i}_transcription.txt')
            
            # Ensure the audio data is a numpy array before writing
            # 쓰기 전에 오디오 데이터가 numpy 배열인지 확인합니다.
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)

            # Save the audio file using the decoded data and sampling rate
            # 디코딩된 데이터와 샘플링 레이트를 사용하여 오디오 파일을 저장합니다.
            sf.write(audio_path, audio_data, sampling_rate)

            # Save the transcription file
            # 스크립션 파일을 저장합니다.
            with open(transcription_path, 'w') as transcription_file:
                transcription_file.write(voice['transcription'])

            print(f'Processed {i} audio clip done')  # Print progress
        else:
            print(f"Skipping entry {i} because 'bytes' key is missing.")
    except Exception as e:
        # This block will catch errors if a particular file is corrupted or unreadable
        # 이 블록은 특정 파일이 손상되었거나 읽을 수 없을 경우 오류를 잡아냅니다.
        print(f"Skipping entry {i} due to an error: {e}")

# Process the dataset in parallel using a ThreadPoolExecutor
# ThreadPoolExecutor를 사용하여 데이터셋을 병렬로 처리합니다.
# The number of workers can be adjusted based on your system's capabilities.
# 작업자 수는 시스템 성능에 따라 조정할 수 있습니다.
# We'll use 8 workers as a reasonable default.
# 합리적인 기본값으로 8개의 작업자를 사용하겠습니다.
with ThreadPoolExecutor(max_workers=12) as executor:
    # Use a list to store the futures
    # 퓨처(Future)를 저장할 리스트를 사용합니다.
    futures = []
    
    # Iterate through the streaming dataset and submit tasks to the executor
    # 스트리밍 데이터셋을 반복하며 실행자에게 작업을 제출합니다.
    for i, voice in enumerate(korean_ganyu):
        futures.append(executor.submit(process_voice, voice, i))
    
    # This ensures the main script waits for all tasks to be completed.
    # 이렇게 하면 모든 작업이 완료될 때까지 메인 스크립트가 기다립니다.
    for future in futures:
        future.result()
