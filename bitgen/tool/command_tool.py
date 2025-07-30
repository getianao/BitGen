import subprocess
import shlex
import os
from ..log import MyLogger

BS_HOME = os.environ.get("BS_HOME")

def exe_command(command, check=True, shell=True, verbose=True):
    if not shell:
        if isinstance(command, str):
            command = shlex.split(command)
        # command = [c.replace("${BS_HOME}", BS_HOME) for c in command]
        if verbose:
            command_str = " ".join(command)
            MyLogger.debug(f"Execute: {command_str}")
    else:
        if isinstance(command, list):
            command = " ".join(command)
        # command = command.replace("${BS_HOME}", BS_HOME)
        if verbose:
            MyLogger.debug(f"Execute: {command}")
            
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=check,
        shell=shell,
        env=os.environ,
    )
    return result