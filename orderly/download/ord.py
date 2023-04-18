from typing import List, Optional
import platform
import subprocess
import os
import logging

LOG = logging.getLogger(__name__)


def run_docker_download(
    output_control: bool,
    build_image: str,
    run_image: str,
    extra_commands: Optional[List[str]] = None,
) -> None:
    with open("/tmp/output.log", "a") as output:
        output_control_dict = {}
        if output_control:
            output_control_dict = {"stdout": output, "stderr": output}

        output_control_dict = {"shell": True, **output_control_dict}  # type: ignore

        subprocess.call(build_image, **output_control_dict)  # type: ignore
        subprocess.call(run_image, **output_control_dict)  # type: ignore

        if extra_commands is not None:
            for i in extra_commands:
                subprocess.call(i, **output_control_dict)  # type: ignore


def linux_download(output_control: bool = True) -> None:
    run_docker_download(
        output_control=output_control,
        build_image="docker image build --target orderly_download_linux --tag orderly_download_linux .",
        run_image=f"docker run -v {os.getcwd()}/data:/tmp_data -u {os.getuid()}:{os.getgid()} -it orderly_download_linux",
    )


def mac_download(output_control: bool = True) -> None:
    run_docker_download(
        output_control=output_control,
        build_image="docker image build --target orderly_download_linux --tag orderly_download_sudo .",
        run_image=f"docker run -v {os.getcwd()}/data:/tmp_data -u {os.getuid()}:{os.getgid()} -it orderly_download_linux",
        extra_commands=[f"sudo chown -R {os.getuid()}:{os.getgid()} {os.getcwd()}"],
    )


def download(output_control: bool = True) -> None:
    if platform.system() == "Windows":
        e = NotImplementedError()
        LOG.error(e)
        raise e
    elif platform.system() == "Linux":
        linux_download(output_control=output_control)
        return
    elif platform.system() == "MacOS":
        mac_download(output_control=output_control)
    else:
        e = NotImplementedError()
        LOG.error(e)
        raise e


if __name__ == "__main__":
    download(False)
