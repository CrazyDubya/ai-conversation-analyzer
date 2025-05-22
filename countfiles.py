import os


def count_files(directory):
    """Count the number of files in the given directory."""
    try:
        # List all entries in the directory
        entries = os.listdir(directory)

        # Filter out directories, count only files
        file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(directory, entry)))

        return file_count
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return None
    except PermissionError:
        print(f"Error: Permission denied to access directory '{directory}'.")
        return None


# Specify the directory path
directory_path = "/Users/puppuccino/PycharmProjects/inner_mon/GPTLOG-2024-05-17-02-12-23/analyzed_conversations-2"

# Count the files
num_files = count_files(directory_path)

if num_files is not None:
    print(f"Number of files in {directory_path}: {num_files}")
