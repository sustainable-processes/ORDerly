import typing
import platform
import subprocess
import os


def run_docker_download(
    output_control,
    build_image: str,
    run_image: str,
    extra_commands: typing.Optional[typing.List[str]] = None,
):
    with open("/tmp/output.log", "a") as output:
        if output_control:
            output_control = {"stdout": output, "stderr": output}
        else:
            output_control = {}
        output_control = {"shell": True, **output_control}

        subprocess.call(build_image, **output_control)
        subprocess.call(run_image, **output_control)

        if extra_commands is not None:
            for i in extra_commands:
                subprocess.call(i, **output_control)


def linux_download(output_control: bool = True):
    run_docker_download(
        output_control=output_control,
        build_image="docker image build --target orderly_download_linux --tag orderly_download_linux .",
        run_image=f"docker run -v {os.getcwd()}/data:/tmp_data -u {os.getuid()}:{os.getgid()} -it orderly_download_linux",
    )


def mac_download(output_control: bool = True):
    run_docker_download(
        output_control=output_control,
        build_image="docker image build --target orderly_download_linux --tag orderly_download_sudo .",
        run_image=f"docker run -v {os.getcwd()}/data:/tmp_data -u {os.getuid()}:{os.getgid()} -it orderly_download_linux",
        extra_commands=[f"sudo chown -R {os.getuid()}:{os.getgid()} {os.getcwd()}"],
    )


def download(output_control=True):
    if platform.system() == "Windows":
        raise NotImplementedError()
    elif platform.system() == "Linux":
        linux_download(output_control=output_control)
        return
    elif platform.system() == "MacOS":
        mac_download(output_control=output_control)
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    download(False)
