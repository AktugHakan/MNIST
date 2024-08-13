from datetime import datetime

debug_level = 0

def info(msg: str):
    if debug_level > 1:
        print("INFO " + _get_current_time_string() + ": " + msg)

def warning(msg: str):
    if debug_level > 0:
        print('\033[93m' + "WARN:" + _get_current_time_string() + ": " + msg + '\033[0m')

def error(msg: str):
    print('\033[91m' + "ERROR:" + _get_current_time_string() + ": " + msg + '\033[0m')
    
def set_debug_level(a: int):
    if not (a in (0,1,2)):
        raise ValueError("debug level must be either 0, 1, 2")
    global debug_level
    debug_level = a

def _get_current_time_string() -> str:
    currentTime = datetime.now()
    return f"({currentTime.hour:02d}:{currentTime.minute:02d}:{currentTime.second})"