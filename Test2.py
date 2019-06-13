import librosa
import numpy as np
from librosa import display
from scipy import signal
import matplotlib.pyplot as plt
from scipy import ndimage

n_fft = 16384
smoothing_size_sec = 2.5
line_threshold = 0.15
min_lines = 8
num_iterations = 16
overlap_percent_margin = 0.2


def loadfile(dir):
    song, sr = librosa.load(dir)
    duration = song.shape[0] / sr
    data = np.abs(librosa.stft(song, n_fft=n_fft))**2
    chroma = librosa.feature.chroma_stft(S=data, sr=sr)
    return chroma, duration


def time_time_similarity_matrix(chroma):
    sample_number = chroma.shape[1]
    time_time_matrix = np.zeros((sample_number, sample_number))
    for i in range(sample_number):
        for j in range(sample_number):
            time_time_matrix[i, j] = 1 - (
                    np.linalg.norm(chroma[:, i] - chroma[:, j]) / np.power(12, 0.5))
    return time_time_matrix


def time_lag_similarity_matrix(chroma):
    sample_number = chroma.shape[1]
    time_lag_matrix = np.zeros((sample_number, sample_number))
    for i in range(sample_number):
        for j in range(i+1):
            time_lag_matrix[j][i] = 1 - (
                    np.linalg.norm(chroma[:, i] - chroma[:, i-j])/np.power(12, 0.5))
    return time_lag_matrix


def denoise(time_lag_matrix, time_time_matrix, smoothing_size):
    n = time_lag_matrix.shape[0]

    horizontal_smoothing_window = np.ones(
        (1, smoothing_size)) / smoothing_size
    horizontal_moving_average = signal.convolve2d(
        time_lag_matrix, horizontal_smoothing_window, mode="full")
    left_average = horizontal_moving_average[:, 0:n]
    right_average = horizontal_moving_average[:, smoothing_size - 1:]
    max_horizontal_average = np.maximum(left_average, right_average)

    vertical_smoothing_window = np.ones((smoothing_size,
                                         1)) / smoothing_size
    vertical_moving_average = signal.convolve2d(
        time_lag_matrix, vertical_smoothing_window, mode="full")
    down_average = vertical_moving_average[0:n, :]
    up_average = vertical_moving_average[smoothing_size - 1:, :]

    diagonal_moving_average = signal.convolve2d(
        time_time_matrix, horizontal_smoothing_window, mode="full")
    ur_average = np.zeros((n, n))
    ll_average = np.zeros((n, n))
    for x in range(n):
        for y in range(x):
            ll_average[y, x] = diagonal_moving_average[x - y, x]
            ur_average[y, x] = diagonal_moving_average[x - y,
                                                       x + smoothing_size - 1]

    non_horizontal_max = np.maximum.reduce([down_average, up_average, ll_average, ur_average])
    non_horizontal_min = np.minimum.reduce([up_average, down_average, ll_average, ur_average])

    suppression = (max_horizontal_average > non_horizontal_max) * non_horizontal_min + (
            max_horizontal_average <= non_horizontal_max) * non_horizontal_max

    denoised_matrix = ndimage.filters.gaussian_filter1d(
        np.triu(time_lag_matrix - suppression), smoothing_size, axis=1)
    denoised_matrix = np.maximum(denoised_matrix, 0)
    denoised_matrix[0:5, :] = 0
    return denoised_matrix


def local_maxima_rows(denoised_time_lag):
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = signal.argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]


def detect_lines(denoised_time_lag, rows, min_length_samples):
    cur_threshold = line_threshold
    line_segments = []
    for _ in range(num_iterations):
        line_segments = detect_lines_helper(denoised_time_lag, rows, cur_threshold, min_length_samples)
        if len(line_segments) >= min_lines:
            return line_segments
        cur_threshold *= 0.95
    return line_segments


def detect_lines_helper(denoised_time_lag, rows, threshold, min_length_samples):
    sample_number = denoised_time_lag.shape[0]
    line_segments = []
    cur_segment_start = None
    for row in rows:
        if row < min_length_samples:
            continue
        for col in range(row, sample_number):
            if denoised_time_lag[row, col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if cur_segment_start is not None and (col - cur_segment_start) > min_length_samples:
                    line_segments.append((cur_segment_start, col, row))
                cur_segment_start = None
    return line_segments


def count_overlapping_lines(lines, margin, min_length_samples):
    line_scores = {}
    for line in lines:
        line_scores[line] = 0

    for line_1 in lines:
        for line_2 in lines:
            lines_overlap_vertically = (
                line_2[0] < (line_1[0] + margin)) and (
                    line_2[1] > (line_1[1] - margin)) and (
                        abs(line_2[2] - line_1[2]) > min_length_samples)

            lines_overlap_diagonally = (
                (line_2[0] - line_2[2]) < (line_1[0] - line_1[2] + margin)) and (
                    (line_2[1] - line_2[2]) > (line_1[1] - line_1[2] - margin)) and (
                        abs(line_2[2] - line_1[2]) > min_length_samples)

            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1
    return line_scores


def best_segment(line_scores):
    lines_to_sort = []
    for line in line_scores:
        lines_to_sort.append((line, line_scores[line], line[1] - line[0]))

    lines_to_sort.sort(key=lambda x: (x[1], x[2]), reverse=True)
    best_tuple = lines_to_sort[0]
    return best_tuple[0]


def find_chorus(chroma, song_length_sec, clip_length):
    sample_number = chroma.shape[1]

    time_time = time_time_similarity_matrix(chroma)
    time_lag = time_lag_similarity_matrix(chroma)

    chroma_sr = sample_number / song_length_sec
    smoothing_size_samples = int(smoothing_size_sec * chroma_sr)

    denoised_lag = denoise(time_lag, time_time, smoothing_size_samples)
    librosa.display.specshow(
        denoised_lag,
        y_axis='time',
        x_axis='time',
        sr=22050 / (n_fft / 2048))
    plt.colorbar()
    plt.set_cmap("hot_r")
    plt.show()

    clip_length_samples = clip_length * chroma_sr
    candidate_rows = local_maxima_rows(denoised_lag)
    lines = detect_lines(denoised_lag, candidate_rows, clip_length_samples)
    if len(lines) == 0:
        print("No choruses were detected. Try a smaller search duration")
        return None
    line_scores = count_overlapping_lines(lines, overlap_percent_margin * clip_length_samples, clip_length_samples)
    best_chorus = best_segment(line_scores)
    return best_chorus[0] / chroma_sr


chroma, dur = loadfile("Test Folder/Live Young Die Free.wav")
print(find_chorus(chroma, dur, 20))
