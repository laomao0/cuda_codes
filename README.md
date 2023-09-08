# cuda_codes

Some coding examples.


Steps to build a cuda program.

1. build and link, nvcc -o xxx xxx.cu
2. run, ./xxx


简单cuda code debug方法

1. 使用vscode远程连接服务器，安装 C/C++ 和 Nsight 插件
2. 创建 launch.json 在 .vscode 文件夹下面，其中 program就是需要debug的可执行文件
3. 使用gdb对cu代码进行build：nvcc -g -G thread.cu -o thread
4. 在cu代码打断点，按F5进行debug





launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "CUDA C++: Launch",
            "type": "cuda-gdb",
            "request": "launch",
            "program": "/home/s00834119/CODES/cuda_codes-master/cuda/thread"
        },
        {
            "name": "CUDA C++: Attach",
            "type": "cuda-gdb",
            "request": "attach"
        }
    ]
}
