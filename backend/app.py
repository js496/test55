from fastapi import FastAPI, Request, HTTPException
import json
import subprocess
import docker
from docker.types import DeviceRequest
import time
import os
import redis.asyncio as redis
import sys
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import pynvml
import logging

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

logger_redis = setup_logger('log_redis', 'logfile_container_redis.log')
logger_redis.info('Logging started!')

logger_backend = setup_logger('log_backend', 'logfile_container_backend.log')
logger_backend.info('Logging started!')

r = redis.Redis(host="redis", port=6379, db=0)

def get_gpu_info():
    req_gpu = 0
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        handle = pynvml.nvmlDeviceGetHandleByIndex(req_gpu)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = memory_info.used / 1024**2
        mem_total = memory_info.total / 1024**2
        mem_util = (mem_used / mem_total) * 100
        gpu_info.append({
                "gpu_count": device_count,
                "gpu_util": float(gpu_util),
                "mem_used": float(mem_used),
                "mem_total": float(mem_total),
                "mem_util": float(mem_util)
        })
        pynvml.nvmlShutdown()     
        return gpu_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logger_backend.exception(f'Exception occured: {e}', exc_info=True)
        return 0

current_gpu_info = get_gpu_info()
gpu_int_arr = [0]

test_gpu_arr = ['0', '1']
test_model = f'nm-testing/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic'
test_port = f'1370'

def start_vllm():
    try:
        res_container = client.containers.run(
            image='flozi00/atra:vllm_main',
            name='vllm',
            environment={
                'env_file': ['.env', '../../.env.global']
            },
            volumes={
                '/volume_hf': {'bind': '/data', 'mode': 'rw'},
                '/mnt/primeline-ai': {'bind': '/mnt/primeline-ai', 'mode': 'rw'},
                '/models': {'bind': '/models', 'mode': 'rw'}
            },
            command=f"--model {test_model} --max-model-len 8192 --cpu-offload-gb 0 --enforce-eager --enable-prefix-caching --tensor-parallel-size 2 --kv-cache-dtype fp8 --port {test_port} --swap-space 2 --gpu-memory-utilization 0.95 --served-model-name deepseek --enable-chunked-prefill --trust-remote-code",
            ports={
                        f'{test_port}/tcp': ("0.0.0.0", test_port)
                    },
            shm_size='32gb',
            restart_policy={"Name": "always"},
            device_requests=[
                docker.types.DeviceRequest(
                    driver='nvidia',
                    device_ids=test_gpu_arr,
                    capabilities=[['gpu']]
                )
            ],
            detach=True,
            auto_remove=False,
            tty=True
        )
        container_id = res_container.id
        test_gpu_str = " ".join(test_gpu_arr)
        redis_add(test_gpu_str,test_model,test_port,test_port,test_port,test_model)
        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logger_backend.exception(f'Exception occured: {e}', exc_info=True)

async def redis_timer():
    while True:
        try:
            current_gpu_info = get_gpu_info()
            res_db_gpu = await r.get('db_gpu')
            if res_db_gpu is not None:
                db_gpu = json.loads(res_db_gpu)
                updated_gpu_data = []
                for gpu_int in range(len(db_gpu)):

                    update_data = {
                        "gpu": gpu_int,
                        "gpu_info": str(current_gpu_info),
                        "running_model": db_gpu[gpu_int].get("running_model", "0"),
                        "timestamp": str(datetime.now()),
                        "port_vllm": db_gpu[gpu_int].get("port_vllm", "0"),
                        "port_model": db_gpu[gpu_int].get("port_model", "0"),
                        "used_ports": db_gpu[gpu_int].get("used_ports", "0"),
                        "used_models": db_gpu[gpu_int].get("used_models", "0"),
                    }
                    updated_gpu_data.append(update_data)
                await r.set('db_gpu', json.dumps(updated_gpu_data))
            else:
                update_data = [{
                    "gpu": 0,
                    "gpu_info": str(current_gpu_info),
                    "running_model": "0",
                    "timestamp": str(datetime.now()),
                    "port_vllm": "0",
                    "port_model": "0",
                    "used_ports": "0",
                    "used_models": "0",
                }]
                await r.set('db_gpu', json.dumps(update_data))
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(0.2)

async def redis_add(gpu,running_model,port_vllm,port_model,used_ports,used_models):
    try:        
        current_gpu_info = get_gpu_info()
        res_db_gpu = await r.get('db_gpu')
        if res_db_gpu is not None:
            db_gpu = json.loads(res_db_gpu)
            add_data = {
                        "gpu": gpu, 
                        "gpu_info": str(current_gpu_info),
                        "running_model": running_model,
                        "timestamp": str(datetime.now()),
                        "port_vllm": port_vllm,
                        "port_model": port_model,
                        "used_ports": used_ports,
                        "used_models": used_models
                        }
            db_gpu += [add_data]
            await r.set('db_gpu', json.dumps(db_gpu))
            await asyncio.sleep(0.2)
        else:
            update_data = {
                "gpu": 0, 
                "gpu_info": str(current_gpu_info),
                "running_model": "0",
                "timestamp": str(datetime.now()),
                "port_vllm": "0",
                "port_model": "0",
                "used_ports": "0",
                "used_models": "0"
                }
            await r.set('db_gpu', json.dumps(update_data))
            await asyncio.sleep(0.2)   
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logger_backend.exception(f'Exception occured: {e}', exc_info=True)
        await asyncio.sleep(0.2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(redis_timer())
    yield

app = FastAPI(lifespan=lifespan)
client = docker.from_env()
device_request = DeviceRequest(count=-1, capabilities=[["gpu"]])

@app.get("/")
async def root():
    return f'Hello from server {os.getenv("CONTAINER_PORT")}!'

@app.post("/dockerrest")
async def docker_rest(request: Request):
    try:
        req_data = await request.json()
                
        if req_data["req_method"] == "logs":
            req_container = client.containers.get(req_data["req_model"])
            res_logs = req_container.logs()
            res_logs_str = res_logs.decode('utf-8')
            return JSONResponse({"result": 200, "result_data": res_logs_str})

        if req_data["req_method"] == "network":
            req_container = client.containers.get(req_data["req_container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result": 200, "result_data": stats})

        if req_data["req_method"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        if req_data["req_method"] == "delete":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "stop":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "start":
            req_container = client.containers.get(req_data["req_model"])
            req_container.start()
            return JSONResponse({"result": 200})




def start_vllm():
    client = docker.from_env()

    container = client.containers.run(
        image='flozi00/atra:vllm_main',
        name='vllm',
        environment={
            'env_file': ['.env', '../../.env.global']
        },
        volumes={
            '/volume_hf': {'bind': '/data', 'mode': 'rw'},
            '/mnt/primeline-ai': {'bind': '/mnt/primeline-ai', 'mode': 'rw'},
            '/models': {'bind': '/models', 'mode': 'rw'}
        },
        command="--model nm-testing/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --max-model-len 8192 --cpu-offload-gb 0 --enforce-eager --enable-prefix-caching --tensor-parallel-size 2 --kv-cache-dtype fp8 --port 1370 --swap-space 2 --gpu-memory-utilization 0.95 --served-model-name deepseek --enable-chunked-prefill --trust-remote-code",
        ports={'1370/tcp': 1370},
        shm_size='32gb',
        restart_policy={"Name": "always"},
        device_requests=[
            docker.types.DeviceRequest(
                driver='nvidia',
                device_ids=['0', '1'],
                capabilities=[['gpu']]
            )
        ],
        detach=True,
        auto_remove=False,
        tty=True
    )
    print(f"Container {container.name} created and started successfully!")


        if req_data["req_method"] == "command":
            req_model = client.containers.get(req_data["req_model"])
            req_name = client.containers.get(req_data["req_model"])
            
            
            req_container = client.containers.get(req_data["req_model"])
            req_container = client.containers.get(req_data["req_model"])
            req_container = client.containers.get(req_data["req_model"])
            req_container.start()
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "create":
            try:
                container_name = str(req_data["req_model"]).replace('/', '_')
                res_db_gpu = await r.get('db_gpu')
                if res_db_gpu is not None:
                    db_gpu = json.loads(res_db_gpu)                    
                                        
                    # check if model already downloaded/downloading
                    all_used_models = [g["used_models"] for g in db_gpu]
                    print(f'all_used_models {all_used_models}')
                    if req_data["req_model"] in all_used_models:
                        return JSONResponse({"result": 302, "result_data": "Model already downloaded. Trying to start container ..."})
                    
                    # check if ports already used
                    all_used_ports = [g["used_ports"] for g in db_gpu]
                    print(f'all_used_ports {all_used_ports}')
                    if req_data["req_port_vllm"] in all_used_ports or req_data["req_port_model"] in all_used_ports:
                        return JSONResponse({"result": 409, "result_data": "Error: Port already in use"})
                    
                    # check if memory available
                    current_gpu_info = get_gpu_info()
                    if current_gpu_info[0]["mem_util"] > 80:
                        all_running_models = [g["running_model"] for g in db_gpu]
                        print(f'all_running_models {all_running_models}')
                        for running_model in all_running_models:
                            req_container = client.containers.get(req_data["req_model"])
                            req_container.stop()
                        
                    
                    # get all used ports
                    all_used_ports += [req_data["req_port_vllm"],req_data["req_port_model"]]
                    all_used_models += [req_data["req_port_model"]]
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now()),
                        "port_vllm": req_data["req_port_vllm"],
                        "port_model": req_data["req_port_model"],
                        "used_ports": str(all_used_ports),
                        "used_models": str(all_used_models)
                    }
                    
                    db_gpu += [add_data]
                    await r.set('db_gpu', json.dumps(db_gpu))                
                
                else:
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now()),
                        "port_vllm": str(req_data["req_port_vllm"]),
                        "port_model": str(req_data["req_port_model"]),
                        "used_ports": f'{str(req_data["req_port_vllm"])},{str(req_data["req_port_model"])}',
                        "used_models": str(str(req_data["req_model"]))
                    }
                    await r.set('db_gpu', json.dumps(add_data))
                        
                                
                res_container = client.containers.run(
                    "vllm/vllm-openai:latest",
                    command=f'--model {req_data["req_model"]}',
                    name=container_name,
                    runtime=req_data["req_runtime"],
                    volumes={"/home/cloud/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"}},
                    environment={"HUGGING_FACE_HUB_TOKEN": os.getenv("HFTK")},
                    ports={
                        f'{req_data["req_port_vllm"]}/tcp': ("0.0.0.0", req_data["req_port_model"])
                    },
                    ipc_mode="host",
                    device_requests=[device_request],
                    detach=True
                )
                container_id = res_container.id
                return JSONResponse({"result": 200, "result_data": str(container_id)})

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                logger_backend.exception(f'Exception occured: {e}', exc_info=True)
                r.delete(f'running_model:{str(req_data["req_model"])}')
                return JSONResponse({"result": 404, "result_data": str(e)})

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        logger_backend.exception(f'Exception occured: {e}', exc_info=True)
        return JSONResponse({"result": 500, "result_data": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTAINER_PORT")))