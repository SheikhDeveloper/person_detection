from process_video import process_video
import sys

if __name__ == '__main__':
    if len(
            sys.argv) != 3 or sys.argv[1][-4:] != '.mp4' or sys.argv[2][-4:] != '.mp4':
        print(
            'Usage: python main.py <path_to_input_video> <path_to_output_video>',
            file=sys.stderr,
            flush=True)
        sys.exit(1)
    process_video(sys.argv[1], sys.argv[2])
