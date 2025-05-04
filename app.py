import gradio as gr

from llm_client.get_llm_client import get_client
from llm_client.orchestra_agent import plan_and_execution

client = get_client()

css = """
footer {display: none !important}
.gradio-container {
    max-width: 1200px;
    margin: auto;
}
.contain {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
}
.submit-btn {
    background: linear-gradient(90deg, #4B79A1 0%, #283E51 100%) !important;
    border: none !important;
    color: white !important;
}
.submit-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}
.title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 1em;
    background: linear-gradient(90deg, #4B79A1 0%, #283E51 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    margin-bottom: 2em;
    color: #666;
    font-size: 24px;
}
.output-image {
    width: 100% !important;
    max-width: 100% !important;
}
"""


def clear_prompt():
    """Function to clear the prompt box."""
    return ""


with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.HTML('<div class="title">CS/DS552 GenAI Final Project - MultiAgent NL2SQL</div>')
    gr.HTML('<div style="text-align: center; margin-bottom: 2em; color: #666; font-size: 24px;">Jingni & Vishal</div>')

    with gr.Column():
        prompt = gr.Textbox(
            label="Query",
            placeholder="Describe the question ...",
            lines=1
        )
        with gr.Row():
            generate_btn = gr.Button(
                "🙆 Submit",
                elem_classes=["submit-btn"]
            )
            clear_btn = gr.Button(
                "🙅 Clear",
                elem_classes=["submit-btn"]
            )
    with gr.Row():
        response = gr.HTML(
            '<div class="response" style="margin-top: 10px; font-size: 14px; color: #666;"></div>',
            elem_id="response")

    generate_btn.click(
        fn=plan_and_execution,
        inputs=[gr.State(client), prompt],
        outputs=[response]
    )
    clear_btn.click(
        fn=clear_prompt,  # The function to call
        inputs=[],  # No input required
        outputs=[prompt]  # Clears the prompt
    )


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)