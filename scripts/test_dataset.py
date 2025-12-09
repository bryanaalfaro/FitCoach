from src.fitness_datasets import load_dataset

full_workout_dataset = load_dataset(
    dataset_name="fitcoach-dataset",
    data_root="./data",
    split="train",
    max_num_videos=10,
    shuffle_videos=False,
    eval_mode="full_workout",
)

full_workout_sliding_window_dataset = load_dataset(
    dataset_name="fitcoach-dataset",
    data_root="./data",
    split="train",
    max_num_videos=10,
    shuffle_videos=False,
    eval_mode="full_workout_sliding_window",
)

for i in range(len(full_workout_dataset)):
    full_workout_responses = full_workout_dataset[i]["responses"]
    full_workout_response_timestamps = full_workout_dataset[i]["response_timestamps"]
    full_workout_segments = full_workout_sliding_window_dataset[i]["segments"]
    full_workout_segments_responses = []
    breakpoint()
    for segment in full_workout_segments:
        full_workout_segments_responses.extend(segment["responses"])
    full_workout_segments_response_timestamps = []
    for segment in full_workout_segments:
        full_workout_segments_response_timestamps.extend(segment["response_timestamps"])
    assert full_workout_responses == full_workout_segments_responses
    assert full_workout_response_timestamps == full_workout_segments_response_timestamps
    print("Datasets match")
    