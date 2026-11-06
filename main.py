from pathlib import Path

def find_files(p: Path):
    """
    Finds all files in directory and subdirectories
    """
    # Turns folder into a list of Path files
    all_paths = list(p.iterdir())

    for item in list_of_paths:
        if item.is_file():
            read_file(item)
        elif item.is_dir():
            find_files(item)
        else:
            raise FileNotFoundError("Path links to neither file nor directory") # TEMPORARY RAISED ERROR

def read_file(p: Path):
    """
    Opens and reads file
    """
    try: 
        with open(p, "r") as file:
            parse_file(file)
    except:
        raise FileNotFoundError("File could not be read") # TEMPORARY RAISED ERROR

def parse_file(file: io.TextIOWrapper):
    """
    Parses file and deals with broken HTML
    """
    pass