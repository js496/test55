import gradio as gr
import docker
import os
import yaml

def change_vllm_image(container_name, new_image, model_path, port, gpu_ids="all"):
    client = docker.from_env()

    try:
        container = client.containers.get(container_name)
        container.stop()
        print(f"Container '{container_name}' stopped.")

        print(f"Pulling image: {new_image}")
        client.images.pull(new_image)
        print(f"Image '{new_image}' pulled.")

        model_name = os.path.basename(model_path)

        client.containers.run(
            image=new_image,
            name=container_name,
            runtime="nvidia",
            gpus=gpu_ids,
            ports={f"{port}/tcp": port},
            volumes={model_path: {"bind": "/app/model", "mode": "rw"}},
            command=["--model", f"/app/model/{model_name}"],
            detach=True
        )
        print(f"Container '{container_name}' started with image '{new_image}'.")
        return "vLLM container updated successfully!"

    except docker.errors.NotFound as e:
        return f"Error: Container or image not found: {e}"
    except docker.errors.APIError as e:
        return f"Docker API Error: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# vLLM Image Updater")
        with gr.Row():
            container_name = gr.Textbox(label="Container Name", value="my-vllm-container")
            new_image = gr.Textbox(label="New Image", value="vllm/vllm-openai:latest")
            model_path = gr.Textbox(label="Model Path", value="/path/to/models")
            port = gr.Number(label="Port", value=8000)
            gpu_ids = gr.Textbox(label="GPU IDs", value="all")

        update_button = gr.Button("Update vLLM Container")
        output_message = gr.Textbox(label="Output")

        update_button.click(
            fn=change_vllm_image,
            inputs=[container_name, new_image, model_path, port, gpu_ids],
            outputs=output_message,
        )

    demo.launch()

if __name__ == "__main__":
    gradio_interface()